from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence


ASSISTANT_ONLY_MASKING_RULE_VERSION = (
    "assistant_only_masking_v0.1"
)

IGNORE_INDEX = -100


@dataclass(frozen=True)
class AssistantOnlyTrainingExample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]

    prompt_token_count: int
    supervised_token_count: int
    total_token_count: int

    rule_version: str = (
        ASSISTANT_ONLY_MASKING_RULE_VERSION
    )


@dataclass(frozen=True)
class AssistantOnlyBatch:
    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    labels: List[List[int]]

    prompt_token_counts: List[int]
    supervised_token_counts: List[int]
    total_token_counts: List[int]

    rule_version: str = (
        ASSISTANT_ONLY_MASKING_RULE_VERSION
    )


def _require_messages(
    messages: Sequence[Mapping[str, Any]],
) -> None:
    if not isinstance(
        messages,
        Sequence,
    ):
        raise TypeError(
            "messages must be a sequence."
        )

    if not messages:
        raise ValueError(
            "messages must not be empty."
        )

    assistant_positions = []

    for index, message in enumerate(
        messages
    ):
        if not isinstance(
            message,
            Mapping,
        ):
            raise TypeError(
                "Each message must be a mapping."
            )

        role = message.get(
            "role"
        )

        content = message.get(
            "content"
        )

        if not isinstance(
            role,
            str,
        ):
            raise TypeError(
                "Message role must be a string."
            )

        if not isinstance(
            content,
            str,
        ):
            raise TypeError(
                "Message content must be a string."
            )

        if not content.strip():
            raise ValueError(
                "Message content must not be empty."
            )

        if role == "assistant":
            assistant_positions.append(
                index
            )

    if len(
        assistant_positions
    ) != 1:
        raise ValueError(
            "assistant_only_masking_v0.1 requires "
            "exactly one assistant message."
        )

    if (
        assistant_positions[0]
        !=
        len(messages)
        -
        1
    ):
        raise ValueError(
            "The single assistant message must "
            "be the final message."
        )


def _tokenize_text(
    *,
    tokenizer: Any,
    text: str,
) -> List[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )

    if not isinstance(
        encoded,
        Mapping,
    ):
        raise TypeError(
            "Tokenizer output must be a mapping."
        )

    input_ids = encoded.get(
        "input_ids"
    )

    if (
        not isinstance(
            input_ids,
            list,
        )
        or
        not input_ids
    ):
        raise ValueError(
            "Tokenizer must return a non-empty "
            "input_ids list."
        )

    if not all(
        isinstance(
            token_id,
            int,
        )
        for token_id in input_ids
    ):
        raise TypeError(
            "Tokenizer input_ids must contain "
            "integers only."
        )

    return list(
        input_ids
    )


def build_assistant_only_training_example(
    *,
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    max_sequence_length: int,
) -> AssistantOnlyTrainingExample:
    """
    Build one deterministic supervised training example.

    Policy v0.1:
    - exactly one assistant message;
    - assistant message must be final;
    - prompt is rendered with add_generation_prompt=True;
    - full conversation is rendered with
      add_generation_prompt=False;
    - prompt tokens must be an exact prefix of full tokens;
    - prompt tokens receive label -100;
    - assistant tokens remain supervised;
    - truncation is forbidden.
    """

    if not isinstance(
        max_sequence_length,
        int,
    ):
        raise TypeError(
            "max_sequence_length must be an integer."
        )

    if max_sequence_length <= 0:
        raise ValueError(
            "max_sequence_length must be positive."
        )

    _require_messages(
        messages
    )

    prompt_messages = list(
        messages[
            :-1
        ]
    )

    full_text = (
        tokenizer.apply_chat_template(
            list(
                messages
            ),
            tokenize=False,
            add_generation_prompt=False,
        )
    )

    prompt_text = (
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    if not isinstance(
        full_text,
        str,
    ):
        raise TypeError(
            "Full chat template must return text."
        )

    if not isinstance(
        prompt_text,
        str,
    ):
        raise TypeError(
            "Prompt chat template must return text."
        )

    full_ids = _tokenize_text(
        tokenizer=tokenizer,
        text=full_text,
    )

    prompt_ids = _tokenize_text(
        tokenizer=tokenizer,
        text=prompt_text,
    )

    if (
        len(
            full_ids
        )
        >
        max_sequence_length
    ):
        raise ValueError(
            "Training example exceeds frozen "
            "max_sequence_length. "
            "Silent truncation is forbidden."
        )

    if (
        len(
            prompt_ids
        )
        >=
        len(
            full_ids
        )
    ):
        raise ValueError(
            "Assistant response contains no "
            "supervised tokens."
        )

    if (
        full_ids[
            :len(
                prompt_ids
            )
        ]
        !=
        prompt_ids
    ):
        raise ValueError(
            "Prompt tokenization is not an exact "
            "prefix of the full conversation. "
            "Assistant-only masking cannot be "
            "constructed safely."
        )

    labels = [
        IGNORE_INDEX
    ] * len(
        prompt_ids
    )

    labels.extend(
        full_ids[
            len(
                prompt_ids
            ):
        ]
    )

    if (
        len(
            labels
        )
        !=
        len(
            full_ids
        )
    ):
        raise RuntimeError(
            "Label/input length mismatch."
        )

    supervised_token_count = sum(
        1
        for label in labels
        if label != IGNORE_INDEX
    )

    if supervised_token_count <= 0:
        raise RuntimeError(
            "No supervised assistant tokens found."
        )

    prompt_token_count = len(
        prompt_ids
    )

    total_token_count = len(
        full_ids
    )

    if (
        prompt_token_count
        +
        supervised_token_count
        !=
        total_token_count
    ):
        raise RuntimeError(
            "Prompt/supervision token accounting "
            "is inconsistent."
        )

    return AssistantOnlyTrainingExample(
        input_ids=list(
            full_ids
        ),
        attention_mask=[
            1
        ] * total_token_count,
        labels=labels,
        prompt_token_count=
            prompt_token_count,
        supervised_token_count=
            supervised_token_count,
        total_token_count=
            total_token_count,
    )


def collate_assistant_only_examples(
    *,
    examples: Sequence[
        AssistantOnlyTrainingExample
    ],
    pad_token_id: int,
) -> AssistantOnlyBatch:
    """
    Right-pad assistant-only examples.

    Padding policy:
    - input_ids receive pad_token_id;
    - attention_mask receives 0;
    - labels receive -100;
    - no packed-example semantics are introduced.
    """

    if not examples:
        raise ValueError(
            "examples must not be empty."
        )

    if not isinstance(
        pad_token_id,
        int,
    ):
        raise TypeError(
            "pad_token_id must be an integer."
        )

    maximum_length = max(
        example.total_token_count
        for example in examples
    )

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    prompt_token_counts = []
    supervised_token_counts = []
    total_token_counts = []

    for example in examples:
        if (
            len(
                example.input_ids
            )
            !=
            example.total_token_count
        ):
            raise ValueError(
                "input_ids length does not match "
                "example.total_token_count."
            )

        if (
            len(
                example.attention_mask
            )
            !=
            example.total_token_count
        ):
            raise ValueError(
                "attention_mask length does not "
                "match example.total_token_count."
            )

        if (
            len(
                example.labels
            )
            !=
            example.total_token_count
        ):
            raise ValueError(
                "labels length does not match "
                "example.total_token_count."
            )

        padding_length = (
            maximum_length
            -
            example.total_token_count
        )

        input_ids = list(
            example.input_ids
        )

        attention_mask = list(
            example.attention_mask
        )

        labels = list(
            example.labels
        )

        input_ids.extend(
            [
                pad_token_id
            ]
            *
            padding_length
        )

        attention_mask.extend(
            [
                0
            ]
            *
            padding_length
        )

        labels.extend(
            [
                IGNORE_INDEX
            ]
            *
            padding_length
        )

        if not (
            len(
                input_ids
            )
            ==
            len(
                attention_mask
            )
            ==
            len(
                labels
            )
            ==
            maximum_length
        ):
            raise RuntimeError(
                "Collated batch lengths are inconsistent."
            )

        batch_input_ids.append(
            input_ids
        )

        batch_attention_mask.append(
            attention_mask
        )

        batch_labels.append(
            labels
        )

        prompt_token_counts.append(
            example.prompt_token_count
        )

        supervised_token_counts.append(
            example.supervised_token_count
        )

        total_token_counts.append(
            example.total_token_count
        )

    return AssistantOnlyBatch(
        input_ids=batch_input_ids,
        attention_mask=
            batch_attention_mask,
        labels=batch_labels,
        prompt_token_counts=
            prompt_token_counts,
        supervised_token_counts=
            supervised_token_counts,
        total_token_counts=
            total_token_counts,
    )
