from __future__ import annotations

import json

import pandas as pd

import app.adaptation.greenhouse_final_acceptance_runner_v0_4_v0_4 as runner

from app.adaptation.greenhouse_final_acceptance_runner_v0_4_v0_4 import (
    ALLOWED_PROFILE_SOURCES,
    EXPECTED_RELATIONS,
    GREENHOUSE_DOMAIN,
    MAX_NEW_TOKENS,
    SERIALIZED_PROFILE_FIELDS,
    build_greenhouse_deterministic_profile,
    build_label_blind_prompt_record,
    build_label_blind_user_message,
    build_profile_index,
    parse_generated_output,
    process_generated_token_ids,
    serialize_normalized_profile,
)


def require(
    condition: bool,
    message: str,
) -> None:

    if not condition:
        raise AssertionError(
            message
        )


def synthetic_dataframe(
) -> pd.DataFrame:

    return pd.DataFrame(
        {
            "order_id":
                [
                    "O-1",
                    "O-2",
                    "O-3",
                ],

            "revenue":
                [
                    10.0,
                    20.0,
                    30.0,
                ],

            "customer_country":
                [
                    "FR",
                    "US",
                    "FR",
                ],

            "mystery_field":
                [
                    "alpha",
                    "beta",
                    "gamma",
                ],
        }
    )


def test_deterministic_profile_wrapper(
) -> None:

    profile = (
        build_greenhouse_deterministic_profile(
            dataset_id=
                "synthetic:greenhouse-preflight",
            filename=
                "synthetic.csv",
            dataframe=
                synthetic_dataframe(),
        )
    )


    require(
        [
            column.column
            for column
            in profile.columns
        ]
        ==
        [
            "order_id",
            "revenue",
            "customer_country",
            "mystery_field",
        ],
        "Column order changed.",
    )


    require(
        all(
            column.source
            in
            ALLOWED_PROFILE_SOURCES
            for column
            in profile.columns
        ),
        "Forbidden semantic source reached.",
    )


    print(
        "Deterministic structural/fallback wrapper: PASS"
    )


def test_serializer_is_exact_and_label_blind(
) -> None:

    profile = (
        build_greenhouse_deterministic_profile(
            dataset_id=
                "synthetic:greenhouse-preflight",
            filename=
                "synthetic.csv",
            dataframe=
                synthetic_dataframe(),
        )
    )


    column = profile.columns[
        1
    ]


    serialized = (
        serialize_normalized_profile(
            column
        )
    )


    payload = json.loads(
        serialized
    )


    require(
        list(
            payload.keys()
        )
        ==
        sorted(
            SERIALIZED_PROFILE_FIELDS
        ),
        "Canonical serialized key set changed.",
    )


    forbidden = {
        "dataset_id",
        "filename",
        "column",
        "domain",
        "confidence",
        "source",
        "quantity_rule_version",
        "semantic_rule_version",
        "sample_values",
        "numeric_summary",
        "row_count",
        "missing_ratio",
        "unique_count",
        "same_concept",
        "same_concept_family",
        "same_domain",
        "distinct_variants",
        "compatible_units",
        "derived_gap_compatible",
        "projected_gold_relation",
    }


    require(
        not (
            set(payload)
            &
            forbidden
        ),
        "Forbidden serializer fields leaked.",
    )


    require(
        serialized
        ==
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ),
        "Serializer is not canonical.",
    )


    print(
        "Canonical label-blind serializer: PASS"
    )


def test_prompt_record_is_exact(
) -> None:

    dataset_id = (
        "synthetic:greenhouse-preflight"
    )


    profile = (
        build_greenhouse_deterministic_profile(
            dataset_id=
                dataset_id,
            filename=
                "synthetic.csv",
            dataframe=
                synthetic_dataframe(),
        )
    )


    index = build_profile_index(
        [
            profile
        ]
    )


    case_identity = {
        "case_id":
            "synthetic-case-0001",

        "left_dataset_id":
            dataset_id,

        "right_dataset_id":
            dataset_id,

        "left_column":
            "revenue",

        "right_column":
            "order_id",
    }


    record = (
        build_label_blind_prompt_record(
            case_identity=
                case_identity,
            profile_index=
                index,
        )
    )


    require(
        set(record)
        ==
        {
            "domain",
            "left_metric",
            "left_description",
            "right_metric",
            "right_description",
        },
        "Prompt-record key set changed.",
    )


    require(
        record[
            "domain"
        ]
        ==
        GREENHOUSE_DOMAIN,
        "Greenhouse domain changed.",
    )


    require(
        "synthetic-case-0001"
        not in
        json.dumps(
            record
        ),
        "Case ID leaked into prompt record.",
    )


    require(
        dataset_id
        not in
        json.dumps(
            record
        ),
        "Dataset ID leaked into prompt record.",
    )


    message = (
        build_label_blind_user_message(
            case_identity=
                case_identity,
            profile_index=
                index,
        )
    )


    require(
        "Metric A name: revenue"
        in
        message,
        "Left metric missing from user message.",
    )


    require(
        "Metric B name: order_id"
        in
        message,
        "Right metric missing from user message.",
    )


    require(
        dataset_id
        not in
        message,
        "Dataset ID leaked into user message.",
    )


    require(
        "synthetic-case-0001"
        not in
        message,
        "Case ID leaked into user message.",
    )


    print(
        "Exact label-blind prompt record/message: PASS"
    )


def test_strict_json_parser(
) -> None:

    valid = (
        parse_generated_output(
            decoded_output=(
                '{"relation":"unrelated",'
                '"reason":"These metrics describe clearly different '
                'business concepts and operational quantities."}'
            )
        )
    )


    require(
        valid[
            "strict_json_valid"
        ]
        is True,
        "Valid strict JSON was rejected.",
    )


    require(
        valid[
            "relation"
        ]
        ==
        "unrelated",
        "Valid relation changed.",
    )


    invalid_extra = (
        parse_generated_output(
            decoded_output=(
                '{"relation":"unrelated",'
                '"reason":"These metrics describe clearly different '
                'business concepts and operational quantities.",'
                '"extra":true}'
            )
        )
    )


    require(
        invalid_extra[
            "strict_json_valid"
        ]
        is False,
        "Additional JSON properties were accepted.",
    )


    invalid_relation = (
        parse_generated_output(
            decoded_output=(
                '{"relation":"same",'
                '"reason":"These metrics describe clearly different '
                'business concepts and operational quantities."}'
            )
        )
    )


    require(
        invalid_relation[
            "strict_json_valid"
        ]
        is False,
        "Unknown relation was accepted.",
    )


    require(
        tuple(
            EXPECTED_RELATIONS
        )
        ==
        (
            "same_metric_different_state",
            "same_process_different_stage",
            "related_distinct_metric",
            "unrelated",
            "uncertain",
        ),
        "Relation vocabulary changed.",
    )


    print(
        "Strict JSON parser: PASS"
    )


class SyntheticTokenizer:
    def __init__(
        self,
        decoded: str,
    ) -> None:
        self.decoded = decoded

    def decode(
        self,
        _ids,
        *,
        skip_special_tokens: bool,
    ) -> str:

        require(
            skip_special_tokens
            is False,
            "Special-token decode policy changed.",
        )

        return self.decoded


def test_terminal_eos_processing(
) -> None:

    tokenizer = SyntheticTokenizer(
        (
            '{"relation":"uncertain",'
            '"reason":"The supplied semantic definitions do not '
            'support a sufficiently safe relationship classification."}'
        )
    )


    processed = (
        process_generated_token_ids(
            tokenizer=
                tokenizer,
            generated_token_ids=[
                10,
                20,
                106,
            ],
        )
    )


    require(
        processed[
            "strict_json_valid"
        ]
        is True,
        "Valid generated output was rejected.",
    )


    require(
        processed[
            "terminal_stop_token_id"
        ]
        ==
        106,
        "Terminal end-of-turn token was not recorded.",
    )


    exhausted = (
        process_generated_token_ids(
            tokenizer=
                tokenizer,
            generated_token_ids=list(
                range(
                    MAX_NEW_TOKENS
                )
            ),
        )
    )


    require(
        exhausted[
            "strict_json_valid"
        ]
        is False,
        "Budget-exhausted generation was accepted.",
    )


    require(
        exhausted[
            "generation_budget_exhausted"
        ]
        is True,
        "Budget exhaustion was not recorded.",
    )


    print(
        "Terminal EOS / generation-budget processing: PASS"
    )


def test_duplicate_columns_fail_closed(
) -> None:

    dataframe = pd.DataFrame(
        [
            [
                1,
                2,
            ]
        ],
        columns=[
            "value",
            "value",
        ],
    )


    try:

        build_greenhouse_deterministic_profile(
            dataset_id=
                "synthetic:duplicate",
            filename=
                "duplicate.csv",
            dataframe=
                dataframe,
        )

    except RuntimeError:
        pass

    else:
        raise AssertionError(
            "Duplicate dataframe columns were accepted."
        )


    print(
        "Duplicate-column fail-closed guard: PASS"
    )



class SyntheticPromptTensor:
    def __init__(
        self,
        token_ids,
    ) -> None:

        self.token_ids = list(
            token_ids
        )

        self.ndim = 2

        self.shape = (
            1,
            len(
                self.token_ids
            ),
        )

        self.device = None


    def to(
        self,
        device,
    ):

        self.device = device

        return self


class SyntheticGeneratedTensor:
    def __init__(
        self,
        token_ids,
    ) -> None:

        self.token_ids = list(
            token_ids
        )


    def detach(
        self,
    ):

        return self


    def cpu(
        self,
    ):

        return self


    def tolist(
        self,
    ):

        return list(
            self.token_ids
        )


class SyntheticModelOutput:
    def __init__(
        self,
        *,
        prompt_ids,
        generated_ids,
    ) -> None:

        self.full_ids = (
            list(
                prompt_ids
            )
            +
            list(
                generated_ids
            )
        )


    def __getitem__(
        self,
        key,
    ):

        require(
            isinstance(
                key,
                tuple,
            )
            and
            len(
                key
            )
            ==
            2,
            "Unexpected model output index.",
        )


        row_index, token_slice = key


        require(
            row_index
            ==
            0,
            "Unexpected model output row.",
        )


        require(
            isinstance(
                token_slice,
                slice,
            ),
            "Unexpected token slice.",
        )


        return SyntheticGeneratedTensor(
            self.full_ids[
                token_slice
            ]
        )


class SyntheticGenerationTokenizer:
    def __init__(
        self,
    ) -> None:

        self.prompt_ids = [
            101,
            102,
            103,
            104,
        ]

        self.apply_calls = []

        self.decode_calls = []


    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_tensors,
        return_dict,
    ):

        self.apply_calls.append(
            {
                "messages":
                    messages,

                "tokenize":
                    tokenize,

                "add_generation_prompt":
                    add_generation_prompt,

                "return_tensors":
                    return_tensors,

                "return_dict":
                    return_dict,
            }
        )


        return SyntheticPromptTensor(
            self.prompt_ids
        )


    def decode(
        self,
        ids,
        *,
        skip_special_tokens,
    ) -> str:

        self.decode_calls.append(
            {
                "ids":
                    list(
                        ids
                    ),

                "skip_special_tokens":
                    skip_special_tokens,
            }
        )


        require(
            skip_special_tokens
            is False,
            "Special-token decode policy changed.",
        )


        return (
            '{"relation":"uncertain",'
            '"reason":"The supplied definitions do not establish '
            'a sufficiently safe semantic relationship."}'
        )


class SyntheticGenerationModel:
    def __init__(
        self,
    ) -> None:

        self.device = "synthetic-device"

        self.generate_calls = []


    def generate(
        self,
        **kwargs,
    ):

        self.generate_calls.append(
            kwargs
        )


        input_ids = kwargs[
            "input_ids"
        ]


        return SyntheticModelOutput(
            prompt_ids=
                input_ids.token_ids,

            generated_ids=[
                901,
                902,
                106,
            ],
        )


class SyntheticInferenceMode:
    def __enter__(
        self,
    ):

        return self


    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> bool:

        return False


class SyntheticTorch:
    def __init__(
        self,
    ) -> None:

        self.ones_like_calls = []


    def ones_like(
        self,
        value,
    ):

        self.ones_like_calls.append(
            value
        )

        return (
            "synthetic-attention-mask"
        )


    def inference_mode(
        self,
    ):

        return SyntheticInferenceMode()


def test_prebuilt_user_message_generation_primitive(
) -> None:

    tokenizer = (
        SyntheticGenerationTokenizer()
    )

    model = (
        SyntheticGenerationModel()
    )

    torch_module = (
        SyntheticTorch()
    )


    prebuilt_message = (
        "PREBUILT LABEL-BLIND GREENHOUSE MESSAGE"
    )


    original_builder = (
        runner.build_label_blind_user_message
    )


    def forbidden_builder(
        **_kwargs,
    ):

        raise RuntimeError(
            (
                "New prebuilt-message primitive "
                "must not rebuild the prompt."
            )
        )


    runner.build_label_blind_user_message = (
        forbidden_builder
    )


    try:

        result = (
            runner.generate_label_blind_case_from_user_message(
                model=
                    model,
                tokenizer=
                    tokenizer,
                user_message=
                    prebuilt_message,
                torch_module=
                    torch_module,
            )
        )

    finally:

        runner.build_label_blind_user_message = (
            original_builder
        )


    require(
        len(
            tokenizer.apply_calls
        )
        ==
        1,
        "Chat template must be applied exactly once.",
    )


    apply_call = (
        tokenizer.apply_calls[
            0
        ]
    )


    require(
        apply_call[
            "messages"
        ]
        ==
        [
            {
                "role":
                    "user",
                "content":
                    prebuilt_message,
            }
        ],
        "Prebuilt user message changed.",
    )


    require(
        apply_call[
            "tokenize"
        ]
        is True,
        "tokenize policy changed.",
    )


    require(
        apply_call[
            "add_generation_prompt"
        ]
        is True,
        "Generation-prompt policy changed.",
    )


    require(
        apply_call[
            "return_tensors"
        ]
        ==
        "pt",
        "return_tensors policy changed.",
    )


    require(
        apply_call[
            "return_dict"
        ]
        is False,
        "return_dict policy changed.",
    )


    require(
        len(
            model.generate_calls
        )
        ==
        1,
        "model.generate must execute exactly once.",
    )


    generate_call = (
        model.generate_calls[
            0
        ]
    )


    require(
        generate_call[
            "do_sample"
        ]
        is False,
        "do_sample changed.",
    )


    require(
        generate_call[
            "num_beams"
        ]
        ==
        1,
        "num_beams changed.",
    )


    require(
        generate_call[
            "max_new_tokens"
        ]
        ==
        runner.MAX_NEW_TOKENS,
        "max_new_tokens changed.",
    )


    require(
        generate_call[
            "eos_token_id"
        ]
        ==
        list(
            runner.EOS_TOKEN_IDS
        ),
        "EOS-token policy changed.",
    )


    require(
        generate_call[
            "pad_token_id"
        ]
        ==
        runner.PAD_TOKEN_ID,
        "PAD-token policy changed.",
    )


    require(
        result[
            "strict_json_valid"
        ]
        is True,
        "Synthetic generated JSON rejected.",
    )


    require(
        result[
            "relation"
        ]
        ==
        "uncertain",
        "Synthetic relation changed.",
    )


    require(
        result[
            "prompt_token_count"
        ]
        ==
        4,
        "Prompt token count changed.",
    )


    require(
        result[
            "generated_token_count"
        ]
        ==
        3,
        "Generated token count changed.",
    )


    require(
        result[
            "terminal_stop_token_id"
        ]
        ==
        106,
        "Terminal EOS handling changed.",
    )


    require(
        len(
            torch_module.ones_like_calls
        )
        ==
        1,
        "Attention-mask construction count changed.",
    )


    print(
        "Prebuilt user-message generation primitive: PASS"
    )


def test_existing_generate_label_blind_case_delegates_once(
) -> None:

    original_builder = (
        runner.build_label_blind_user_message
    )

    original_delegate = (
        runner.generate_label_blind_case_from_user_message
    )


    builder_calls = []

    delegate_calls = []


    def synthetic_builder(
        *,
        case_identity,
        profile_index,
    ) -> str:

        builder_calls.append(
            {
                "case_identity":
                    case_identity,

                "profile_index":
                    profile_index,
            }
        )

        return (
            "SYNTHETIC PREBUILT MESSAGE"
        )


    sentinel_result = {
        "strict_json_valid":
            True,

        "relation":
            "uncertain",

        "reason":
            "Synthetic delegated generation preserves the existing wrapper behavior safely.",
    }


    def synthetic_delegate(
        *,
        model,
        tokenizer,
        user_message,
        torch_module,
    ):

        delegate_calls.append(
            {
                "model":
                    model,

                "tokenizer":
                    tokenizer,

                "user_message":
                    user_message,

                "torch_module":
                    torch_module,
            }
        )

        return sentinel_result


    runner.build_label_blind_user_message = (
        synthetic_builder
    )

    runner.generate_label_blind_case_from_user_message = (
        synthetic_delegate
    )


    model = object()

    tokenizer = object()

    torch_module = object()

    case_identity = {
        "left_dataset_id":
            "synthetic:left",

        "right_dataset_id":
            "synthetic:right",

        "left_column":
            "metric_a",

        "right_column":
            "metric_b",
    }

    profile_index = {}


    try:

        result = (
            runner.generate_label_blind_case(
                model=
                    model,
                tokenizer=
                    tokenizer,
                case_identity=
                    case_identity,
                profile_index=
                    profile_index,
                torch_module=
                    torch_module,
            )
        )

    finally:

        runner.build_label_blind_user_message = (
            original_builder
        )

        runner.generate_label_blind_case_from_user_message = (
            original_delegate
        )


    require(
        len(
            builder_calls
        )
        ==
        1,
        "Existing API must build the message exactly once.",
    )


    require(
        len(
            delegate_calls
        )
        ==
        1,
        "Existing API must delegate exactly once.",
    )


    require(
        delegate_calls[
            0
        ][
            "user_message"
        ]
        ==
        "SYNTHETIC PREBUILT MESSAGE",
        "Existing API delegated the wrong user message.",
    )


    require(
        delegate_calls[
            0
        ][
            "model"
        ]
        is
        model,
        "Model identity changed during delegation.",
    )


    require(
        delegate_calls[
            0
        ][
            "tokenizer"
        ]
        is
        tokenizer,
        "Tokenizer identity changed during delegation.",
    )


    require(
        delegate_calls[
            0
        ][
            "torch_module"
        ]
        is
        torch_module,
        "Torch-module identity changed during delegation.",
    )


    require(
        result
        is
        sentinel_result,
        "Existing API changed delegated result.",
    )


    print(
        "Existing generate_label_blind_case delegation compatibility: PASS"
    )

def main(
) -> None:

    print(
        "=== DATALENS QLORA v0.4 GREENHOUSE RUNNER CORE v0.4 ==="
    )

    print()


    test_deterministic_profile_wrapper()

    test_serializer_is_exact_and_label_blind()

    test_prompt_record_is_exact()

    test_strict_json_parser()

    test_terminal_eos_processing()

    test_prebuilt_user_message_generation_primitive()

    test_existing_generate_label_blind_case_delegates_once()

    test_duplicate_columns_fail_closed()


    print()

    print(
        "Protected Greenhouse dataset read: False"
    )

    print(
        "Protected Greenhouse cases read: False"
    )

    print(
        "Protected Greenhouse independence read: False"
    )

    print(
        "Model loaded: False"
    )

    print(
        "Adapter loaded: False"
    )

    print(
        "CUDA requested: False"
    )

    print(
        "Training executed: False"
    )

    print(
        "Optimizer created: False"
    )

    print(
        "Backward executed: False"
    )


    print()

    print(
        "DATALENS QLORA v0.4 GREENHOUSE RUNNER CORE v0.4: PASS"
    )


if __name__ == "__main__":
    main()
