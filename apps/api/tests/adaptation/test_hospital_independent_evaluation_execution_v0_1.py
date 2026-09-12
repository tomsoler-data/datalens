from __future__ import annotations


import ast
import importlib
import json


from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any


MODULE_NAME = (
    "app.adaptation."
    "hospital_independent_evaluation_execution_v0_1"
)


SOURCE_PATH = Path(
    "app/adaptation/"
    "hospital_independent_evaluation_execution_v0_1.py"
)


def load_execution_module() -> Any:

    return importlib.import_module(
        MODULE_NAME
    )


class FakeVector:

    def __init__(
        self,
        values: list[
            int
        ],
    ) -> None:

        self.values = list(
            values
        )


    def detach(
        self,
    ) -> "FakeVector":

        return self


    def cpu(
        self,
    ) -> "FakeVector":

        return self


    def tolist(
        self,
    ) -> list[
        int
    ]:

        return list(
            self.values
        )


class FakeTensor:

    def __init__(
        self,
        rows: list[
            list[
                int
            ]
        ],
    ) -> None:

        if not rows:

            raise ValueError(
                "FakeTensor requires rows."
            )

        width = len(
            rows[
                0
            ]
        )

        if any(
            len(
                row
            )
            !=
            width

            for row
            in rows
        ):

            raise ValueError(
                "FakeTensor rows must have equal width."
            )

        self.rows = [
            list(
                row
            )

            for row
            in rows
        ]

        self.ndim = 2

        self.shape = (
            len(
                self.rows
            ),
            width,
        )

        self.device = "cpu"


    def to(
        self,
        device: str,
    ) -> "FakeTensor":

        self.device = device

        return self


    def __getitem__(
        self,
        key: Any,
    ) -> Any:

        if (
            isinstance(
                key,
                tuple,
            )
            and
            len(
                key
            )
            ==
            2
        ):

            row_index = key[
                0
            ]

            column_selector = key[
                1
            ]

            row = self.rows[
                row_index
            ]

            if isinstance(
                column_selector,
                slice,
            ):

                return FakeVector(
                    row[
                        column_selector
                    ]
                )

        return self.rows[
            key
        ]


class FakeInferenceMode(
    AbstractContextManager[
        None
    ]
):

    def __init__(
        self,
        torch_module: "FakeTorch",
    ) -> None:

        self.torch_module = (
            torch_module
        )


    def __enter__(
        self,
    ) -> None:

        self.torch_module.inference_entries += 1

        return None


    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:

        self.torch_module.inference_exits += 1

        return None


class FakeTorch:

    long = "fake-long"


    def __init__(
        self,
    ) -> None:

        self.inference_entries = 0
        self.inference_exits = 0
        self.ones_like_calls: list[
            dict[
                str,
                Any,
            ]
        ] = []


    def inference_mode(
        self,
    ) -> FakeInferenceMode:

        return FakeInferenceMode(
            self
        )


    def ones_like(
        self,
        tensor: FakeTensor,
        *,
        dtype: Any,
        device: str,
    ) -> FakeTensor:

        self.ones_like_calls.append(
            {
                "dtype":
                    dtype,

                "device":
                    device,

                "shape":
                    tensor.shape,
            }
        )

        return FakeTensor(
            [
                [
                    1
                    for _ in row
                ]

                for row
                in tensor.rows
            ]
        ).to(
            device
        )


class FakeTokenizer:

    def __init__(
        self,
        decode_map: dict[
            tuple[
                int,
                ...
            ],
            str,
        ],
    ) -> None:

        self.decode_map = dict(
            decode_map
        )

        self.template_calls: list[
            dict[
                str,
                Any,
            ]
        ] = []

        self.decode_calls: list[
            dict[
                str,
                Any,
            ]
        ] = []


    def apply_chat_template(
        self,
        messages: list[
            dict[
                str,
                str,
            ]
        ],
        **kwargs: Any,
    ) -> FakeTensor:

        self.template_calls.append(
            {
                "messages":
                    messages,

                "kwargs":
                    dict(
                        kwargs
                    ),
            }
        )

        return FakeTensor(
            [
                [
                    11,
                    12,
                    13,
                ]
            ]
        )


    def decode(
        self,
        token_ids: list[
            int
        ],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:

        key = tuple(
            token_ids
        )

        self.decode_calls.append(
            {
                "token_ids":
                    key,

                "skip_special_tokens":
                    skip_special_tokens,

                "clean_up_tokenization_spaces":
                    clean_up_tokenization_spaces,
            }
        )

        if key not in self.decode_map:

            raise RuntimeError(
                f"Unexpected synthetic decode key: {key}"
            )

        return self.decode_map[
            key
        ]


class FakeConfig:

    max_position_embeddings = 4096


class FakeModel:

    def __init__(
        self,
        *,
        label: str,
        generated_ids: list[
            int
        ],
        call_log: list[
            str
        ],
    ) -> None:

        self.label = label

        self.generated_ids = list(
            generated_ids
        )

        self.call_log = call_log

        self.config = FakeConfig()

        self.generate_calls: list[
            dict[
                str,
                Any,
            ]
        ] = []


    def generate(
        self,
        **kwargs: Any,
    ) -> FakeTensor:

        self.call_log.append(
            self.label
        )

        self.generate_calls.append(
            dict(
                kwargs
            )
        )

        input_ids = kwargs[
            "input_ids"
        ]

        prompt_ids = list(
            input_ids.rows[
                0
            ]
        )

        return FakeTensor(
            [
                prompt_ids
                +
                self.generated_ids
            ]
        )


BASE_JSON = json.dumps(
    {
        "relation":
            "unrelated",

        "reason":
            (
                "These metrics describe separate "
                "operational processes without a "
                "direct shared measurement."
            ),
    },
    separators=(
        ",",
        ":",
    ),
)


ADAPTED_JSON = json.dumps(
    {
        "relation":
            "same_process_different_stage",

        "reason":
            (
                "These metrics belong to one "
                "operational process but represent "
                "different sequential stages."
            ),
    },
    separators=(
        ",",
        ":",
    ),
)


def synthetic_inputs() -> tuple[
    dict[
        str,
        str,
    ],
    ...,
]:

    return (
        {
            "case_id":
                "synthetic-001",

            "prompt":
                "Synthetic prompt alpha.",
        },
        {
            "case_id":
                "synthetic-002",

            "prompt":
                "Synthetic prompt beta.",
        },
    )


def assert_raises(
    exception_type: type[
        BaseException
    ],
    function: Any,
) -> None:

    try:

        function()

    except exception_type:

        return

    raise AssertionError(
        (
            "Expected exception was not raised: "
            f"{exception_type.__name__}"
        )
    )


def test_frozen_generation_constants() -> None:

    execution = load_execution_module()

    assert (
        execution.MAX_NEW_TOKENS
        ==
        64
    )

    assert (
        execution.EOS_TOKEN_IDS
        ==
        (
            1,
            106,
        )
    )

    assert (
        execution.PAD_TOKEN_ID
        ==
        0
    )


def test_execution_input_exact_boundary() -> None:

    execution = load_execution_module()

    good = execution.validate_execution_input(
        {
            "case_id":
                "synthetic-001",

            "prompt":
                "Synthetic prompt.",
        }
    )

    assert set(
        good
    ) == {
        "case_id",
        "prompt",
    }


    assert_raises(
        RuntimeError,
        lambda:
            execution.validate_execution_input(
                {
                    "case_id":
                        "synthetic-001",

                    "prompt":
                        "Synthetic prompt.",

                    "gold_relation":
                        "SENTINEL",
                }
            ),
    )


    assert_raises(
        RuntimeError,
        lambda:
            execution.validate_execution_input(
                {
                    "case_id":
                        "synthetic-001",

                    "prompt":
                        "Synthetic prompt.",

                    "gold_reason":
                        "SENTINEL",
                }
            ),
    )


def test_strict_json_parser() -> None:

    execution = load_execution_module()


    valid = execution.parse_generated_output(
        decoded_output=
            ADAPTED_JSON,

        terminal_stop_token_id=
            1,
    )

    assert (
        valid[
            "strict_json_valid"
        ]
        is True
    )

    assert (
        valid[
            "predicted_relation"
        ]
        ==
        "same_process_different_stage"
    )


    malformed = execution.parse_generated_output(
        decoded_output=
            "{not-json}",
    )

    assert (
        malformed[
            "invalid_reason"
        ]
        ==
        "json_parse_failed"
    )


    extra_key = execution.parse_generated_output(
        decoded_output=
            json.dumps(
                {
                    "relation":
                        "unrelated",

                    "reason":
                        (
                            "These metrics belong to "
                            "different operational "
                            "processes with no direct "
                            "shared measurement."
                        ),

                    "extra":
                        True,
                }
            )
    )

    assert (
        extra_key[
            "invalid_reason"
        ]
        ==
        "json_key_set_mismatch"
    )


    bad_relation = execution.parse_generated_output(
        decoded_output=
            json.dumps(
                {
                    "relation":
                        "invented_label",

                    "reason":
                        (
                            "This synthetic explanation "
                            "contains enough words for "
                            "the strict parser contract."
                        ),
                }
            )
    )

    assert (
        bad_relation[
            "invalid_reason"
        ]
        ==
        "relation_not_allowed"
    )


    short_reason = execution.parse_generated_output(
        decoded_output=
            json.dumps(
                {
                    "relation":
                        "unrelated",

                    "reason":
                        "Too few words.",
                }
            )
    )

    assert (
        short_reason[
            "invalid_reason"
        ]
        ==
        "reason_word_count_out_of_range"
    )


    exhausted = execution.parse_generated_output(
        decoded_output=
            ADAPTED_JSON,

        generation_budget_exhausted=
            True,
    )

    assert (
        exhausted[
            "invalid_reason"
        ]
        ==
        "generation_budget_exhausted"
    )


def test_generated_token_processing() -> None:

    execution = load_execution_module()

    tokenizer = FakeTokenizer(
        {
            (
                201,
            ):
                ADAPTED_JSON,
        }
    )


    processed = execution.process_generated_token_ids(
        tokenizer=
            tokenizer,

        generated_token_ids=
            [
                201,
                1,
            ],
    )

    assert (
        processed[
            "strict_json_valid"
        ]
        is True
    )

    assert (
        processed[
            "terminal_stop_token_id"
        ]
        ==
        1
    )


    missing_stop = execution.process_generated_token_ids(
        tokenizer=
            tokenizer,

        generated_token_ids=
            [
                201,
            ],
    )

    assert (
        missing_stop[
            "invalid_reason"
        ]
        ==
        "missing_terminal_stop_token"
    )


    assert_raises(
        RuntimeError,
        lambda:
            execution.process_generated_token_ids(
                tokenizer=
                    tokenizer,

                generated_token_ids=
                    [
                        201
                        for _
                        in range(
                            65
                        )
                    ],
            ),
    )


def test_single_generation_is_deterministic() -> None:

    execution = load_execution_module()

    torch_module = FakeTorch()

    tokenizer = FakeTokenizer(
        {
            (
                202,
            ):
                ADAPTED_JSON,
        }
    )

    log: list[
        str
    ] = []

    model = FakeModel(
        label=
            "synthetic",

        generated_ids=
            [
                202,
                106,
            ],

        call_log=
            log,
    )


    result = execution.generate_execution_input(
        model=
            model,

        tokenizer=
            tokenizer,

        execution_input=
            {
                "case_id":
                    "synthetic-001",

                "prompt":
                    "Synthetic deterministic prompt.",
            },

        torch_module=
            torch_module,
    )


    assert (
        result[
            "case_id"
        ]
        ==
        "synthetic-001"
    )

    assert (
        result[
            "strict_json_valid"
        ]
        is True
    )


    assert log == [
        "synthetic"
    ]


    assert len(
        model.generate_calls
    ) == 1


    kwargs = model.generate_calls[
        0
    ]


    assert (
        kwargs[
            "do_sample"
        ]
        is False
    )

    assert (
        kwargs[
            "num_beams"
        ]
        ==
        1
    )

    assert (
        kwargs[
            "max_new_tokens"
        ]
        ==
        64
    )

    assert (
        kwargs[
            "eos_token_id"
        ]
        ==
        [
            1,
            106,
        ]
    )

    assert (
        kwargs[
            "pad_token_id"
        ]
        ==
        0
    )


    template_kwargs = tokenizer.template_calls[
        0
    ][
        "kwargs"
    ]


    assert (
        template_kwargs[
            "tokenize"
        ]
        is True
    )

    assert (
        template_kwargs[
            "add_generation_prompt"
        ]
        is True
    )

    assert (
        template_kwargs[
            "truncation"
        ]
        is False
    )

    assert (
        template_kwargs[
            "return_tensors"
        ]
        ==
        "pt"
    )

    assert (
        template_kwargs[
            "return_dict"
        ]
        is False
    )


    assert (
        torch_module.inference_entries
        ==
        1
    )

    assert (
        torch_module.inference_exits
        ==
        1
    )


def test_base_then_adapted_order_and_prompt_identity() -> None:

    execution = load_execution_module()

    call_log: list[
        str
    ] = []

    torch_module = FakeTorch()

    tokenizer = FakeTokenizer(
        {
            (
                201,
            ):
                BASE_JSON,

            (
                202,
            ):
                ADAPTED_JSON,
        }
    )

    base_model = FakeModel(
        label=
            "base",

        generated_ids=
            [
                201,
                1,
            ],

        call_log=
            call_log,
    )

    adapted_model = FakeModel(
        label=
            "adapted",

        generated_ids=
            [
                202,
                1,
            ],

        call_log=
            call_log,
    )


    result = execution.run_base_then_adapted(
        base_model=
            base_model,

        adapted_model=
            adapted_model,

        tokenizer=
            tokenizer,

        execution_inputs=
            synthetic_inputs(),

        torch_module=
            torch_module,
    )


    assert (
        result[
            "execution_order"
        ]
        ==
        (
            "base",
            "adapted",
        )
    )


    assert call_log == [
        "base",
        "base",
        "adapted",
        "adapted",
    ]


    assert tuple(
        item[
            "case_id"
        ]

        for item
        in result[
            "base"
        ]
    ) == (
        "synthetic-001",
        "synthetic-002",
    )


    assert tuple(
        item[
            "case_id"
        ]

        for item
        in result[
            "adapted"
        ]
    ) == (
        "synthetic-001",
        "synthetic-002",
    )


    assert tuple(
        item[
            "prompt_sha256"
        ]

        for item
        in result[
            "base"
        ]
    ) == tuple(
        item[
            "prompt_sha256"
        ]

        for item
        in result[
            "adapted"
        ]
    )


    assert all(
        item[
            "model_label"
        ]
        ==
        "base"

        for item
        in result[
            "base"
        ]
    )


    assert all(
        item[
            "model_label"
        ]
        ==
        "adapted"

        for item
        in result[
            "adapted"
        ]
    )


def test_execution_module_has_no_gold_or_holdout_authority() -> None:

    source = SOURCE_PATH.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )


    forbidden_source_tokens = (
        "gold_relation",
        "gold_reason",
        "HOLDOUT",
        "holdout",
        "load_frozen_holdout",
        "from_pretrained",
    )


    for token in forbidden_source_tokens:

        assert token not in source


    heavy_prefixes = (
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
    )


    for node in tree.body:

        if isinstance(
            node,
            ast.Import,
        ):

            for alias in node.names:

                assert not any(
                    alias.name.startswith(
                        prefix
                    )

                    for prefix
                    in heavy_prefixes
                )


        elif isinstance(
            node,
            ast.ImportFrom,
        ):

            if node.module:

                assert not any(
                    node.module.startswith(
                        prefix
                    )

                    for prefix
                    in heavy_prefixes
                )


def main() -> None:

    tests = (
        test_frozen_generation_constants,
        test_execution_input_exact_boundary,
        test_strict_json_parser,
        test_generated_token_processing,
        test_single_generation_is_deterministic,
        test_base_then_adapted_order_and_prompt_identity,
        test_execution_module_has_no_gold_or_holdout_authority,
    )


    print(
        "=== R8 HOSPITAL SYNTHETIC EXECUTION "
        "PLUMBING v0.1 ==="
    )

    print()


    for test in tests:

        test()

        print(
            f"[PASS] {test.__name__}"
        )


    print()

    print(
        "PASS - hospital synthetic execution "
        "plumbing v0.1"
    )


if __name__ == "__main__":

    main()
