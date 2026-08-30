from __future__ import annotations

import json

import pandas as pd

from app.adaptation.greenhouse_final_acceptance_runner_v0_4_v0_2 import (
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


def main(
) -> None:

    print(
        "=== DATALENS QLORA v0.4 GREENHOUSE RUNNER CORE v0.2 ==="
    )

    print()


    test_deterministic_profile_wrapper()

    test_serializer_is_exact_and_label_blind()

    test_prompt_record_is_exact()

    test_strict_json_parser()

    test_terminal_eos_processing()

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
        "DATALENS QLORA v0.4 GREENHOUSE RUNNER CORE v0.2: PASS"
    )


if __name__ == "__main__":
    main()
