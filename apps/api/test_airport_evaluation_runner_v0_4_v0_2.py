from __future__ import annotations

import ast
from pathlib import Path

from app.adaptation.airport_evaluation_runner_v0_4_v0_2 import (
    EOS_TOKEN_IDS,
    EXPECTED_PROTOCOL_GIT_COMMIT,
    EXPECTED_PROTOCOL_SHA256,
    EXPECTED_RELATIONS,
    MAX_NEW_TOKENS,
    evaluate_acceptance_gates,
    paired_comparison,
    parse_generated_output,
    process_generated_token_ids,
)


print(
    "=== DATALENS QLORA v0.4 AIRPORT EVALUATION RUNNER TEST v0.2 ==="
)


class FakeTokenizer:
    def __init__(
        self,
        mapping: dict[
            tuple[
                int,
                ...,
            ],
            str,
        ],
    ) -> None:
        self.mapping = mapping

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        if (
            skip_special_tokens
            is not False
        ):
            raise AssertionError(
                (
                    "skip_special_tokens "
                    "must stay False"
                )
            )

        if (
            clean_up_tokenization_spaces
            is not False
        ):
            raise AssertionError(
                (
                    "decode cleanup "
                    "must stay False"
                )
            )

        return self.mapping.get(
            tuple(
                int(
                    value
                )
                for value
                in token_ids
            ),
            "",
        )


def valid_json(
    relation: str = "related_distinct_metric",
) -> str:
    return (
        '{"relation":"'
        +
        relation
        +
        '","reason":"Both supplied definitions describe related '
        'operational quantities while measuring distinct events '
        'in the workflow."}'
    )


def case_result(
    *,
    case_id: str,
    expected: str,
    predicted: (
        str
        | None
    ),
    valid: bool = True,
) -> dict:
    return {
        "case_id":
            case_id,

        "expected_relation":
            expected,

        "predicted_relation":
            predicted,

        "correct":
            bool(
                valid
                and
                predicted
                ==
                expected
            ),

        "strict_json_valid":
            valid,
    }


def model_result(
    *,
    relation_correct_counts: dict[
        str,
        int,
    ],
    valid_count: int = 30,
) -> dict:
    cases = []

    for relation in EXPECTED_RELATIONS:
        correct_for_relation = (
            relation_correct_counts[
                relation
            ]
        )

        for index in range(
            6
        ):
            correct = (
                index
                <
                correct_for_relation
            )

            predicted = (
                relation
                if correct
                else next(
                    candidate
                    for candidate
                    in EXPECTED_RELATIONS
                    if candidate
                    !=
                    relation
                )
            )

            cases.append(
                case_result(
                    case_id=
                        f"{relation}:{index + 1}",

                    expected=
                        relation,

                    predicted=
                        predicted,

                    valid=
                        True,
                )
            )

    if valid_count < 30:
        invalid_needed = (
            30
            -
            valid_count
        )

        for item in cases:
            if invalid_needed <= 0:
                break

            item[
                "strict_json_valid"
            ] = False

            item[
                "predicted_relation"
            ] = None

            item[
                "correct"
            ] = False

            invalid_needed -= 1

    correct_total = sum(
        1
        for item
        in cases
        if item[
            "correct"
        ]
    )

    per_relation = {}

    for relation in EXPECTED_RELATIONS:
        relation_cases = [
            item
            for item
            in cases
            if item[
                "expected_relation"
            ]
            ==
            relation
        ]

        per_relation[
            relation
        ] = round(
            (
                sum(
                    1
                    for item
                    in relation_cases
                    if item[
                        "correct"
                    ]
                )
                /
                6
            ),
            6,
        )

    macro = round(
        (
            sum(
                per_relation.values()
            )
            /
            len(
                EXPECTED_RELATIONS
            )
        ),
        6,
    )

    return {
        "case_count":
            30,

        "correct_count":
            correct_total,

        "strict_json_valid_count":
            valid_count,

        "accuracy":
            round(
                correct_total
                /
                30,
                6,
            ),

        "macro_accuracy":
            macro,

        "strict_json_validity_rate":
            round(
                valid_count
                /
                30,
                6,
            ),

        "per_relation_accuracy":
            per_relation,

        "uncertain_accuracy":
            per_relation[
                "uncertain"
            ],

        "cases":
            cases,
    }


# ============================================================
# FROZEN IDENTITIES
# ============================================================


assert (
    EXPECTED_PROTOCOL_GIT_COMMIT
    ==
    "f6b8063808badc05a4cc93abe80323d1d745f17d"
)

assert (
    EXPECTED_PROTOCOL_SHA256
    ==
    (
        "a88cb86565c0a3d4daa4d5b2bea3beee"
        "d79a7df81155ed4d4ed0c7bfef2e00d1"
    )
)

assert MAX_NEW_TOKENS == 64

assert EOS_TOKEN_IDS == (
    1,
    106,
)

assert EXPECTED_RELATIONS == (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
)


print(
    "[PASS] frozen Airport protocol identity"
)


# ============================================================
# STRICT JSON PARSER
# ============================================================


parsed = parse_generated_output(
    decoded_output=
        valid_json(),
)

assert (
    parsed[
        "strict_json_valid"
    ]
    is True
)

assert (
    parsed[
        "predicted_relation"
    ]
    ==
    "related_distinct_metric"
)

assert (
    parsed[
        "reason_word_count"
    ]
    >=
    6
)


assert (
    parse_generated_output(
        decoded_output=
            (
                '{"relation":"unrelated",'
                '"reason":"too short"}'
            ),
    )[
        "invalid_reason"
    ]
    ==
    "reason_word_count_out_of_range"
)


assert (
    parse_generated_output(
        decoded_output=
            (
                '{"relation":"unrelated",'
                '"reason":"These definitions describe different '
                'operational quantities and do not establish '
                'a shared process or common measured state.",'
                '"extra":true}'
            ),
    )[
        "invalid_reason"
    ]
    ==
    "json_key_set_mismatch"
)


assert (
    parse_generated_output(
        decoded_output=
            (
                valid_json()
                +
                "\ntrailing prose"
            ),
    )[
        "invalid_reason"
    ]
    ==
    "json_parse_failed"
)


assert (
    parse_generated_output(
        decoded_output=
            (
                '{"relation":"not_a_relation",'
                '"reason":"The supplied definitions do not '
                'support this invented relationship label '
                'in the frozen five class task."}'
            ),
    )[
        "invalid_reason"
    ]
    ==
    "relation_not_allowed"
)


assert (
    parse_generated_output(
        decoded_output=
            '["relation","unrelated"]',
    )[
        "invalid_reason"
    ]
    ==
    "json_value_not_object"
)


assert (
    parse_generated_output(
        decoded_output=
            valid_json(),

        generation_budget_exhausted=
            True,
    )[
        "invalid_reason"
    ]
    ==
    "generation_budget_exhausted"
)


print(
    "[PASS] strict JSON parser fail-closed surface"
)


# ============================================================
# TERMINAL EOS / EOT HANDLING
# ============================================================


text = valid_json()

tokenizer = FakeTokenizer(
    {
        (
            10,
            11,
            12,
        ):
            text,

        (
            10,
            11,
            12,
            106,
        ):
            (
                text
                +
                "<end_of_turn>"
            ),
    }
)


processed_eos = (
    process_generated_token_ids(
        tokenizer=
            tokenizer,

        generated_token_ids=
            [
                10,
                11,
                12,
                1,
            ],
    )
)

assert (
    processed_eos[
        "strict_json_valid"
    ]
    is True
)

assert (
    processed_eos[
        "terminal_stop_token_id"
    ]
    ==
    1
)


processed_eot = (
    process_generated_token_ids(
        tokenizer=
            tokenizer,

        generated_token_ids=
            [
                10,
                11,
                12,
                106,
            ],
    )
)

assert (
    processed_eot[
        "strict_json_valid"
    ]
    is True
)

assert (
    processed_eot[
        "terminal_stop_token_id"
    ]
    ==
    106
)


# Exactly one terminal special token is removed.
# The second remains visible to decode and must fail JSON.
double_stop = (
    process_generated_token_ids(
        tokenizer=
            tokenizer,

        generated_token_ids=
            [
                10,
                11,
                12,
                106,
                106,
            ],
    )
)

assert (
    double_stop[
        "strict_json_valid"
    ]
    is False
)

assert (
    double_stop[
        "invalid_reason"
    ]
    ==
    "json_parse_failed"
)


budget_tokenizer = FakeTokenizer(
    {
        tuple(
            range(
                MAX_NEW_TOKENS
            )
        ):
            text,
    }
)

budget = (
    process_generated_token_ids(
        tokenizer=
            budget_tokenizer,

        generated_token_ids=
            list(
                range(
                    MAX_NEW_TOKENS
                )
            ),
    )
)

assert (
    budget[
        "strict_json_valid"
    ]
    is False
)

assert (
    budget[
        "invalid_reason"
    ]
    ==
    "generation_budget_exhausted"
)


missing_stop = (
    process_generated_token_ids(
        tokenizer=
            tokenizer,

        generated_token_ids=
            [
                10,
                11,
                12,
            ],
    )
)

assert (
    missing_stop[
        "strict_json_valid"
    ]
    is False
)

assert (
    missing_stop[
        "invalid_reason"
    ]
    ==
    "missing_terminal_stop_token"
)


print(
    "[PASS] EOS/EOT visibility and generation-budget enforcement"
)


# ============================================================
# PAIRED METRICS / GATES
# ============================================================


base = model_result(
    relation_correct_counts={
        relation:
            4
        for relation
        in EXPECTED_RELATIONS
    }
)

adapted = model_result(
    relation_correct_counts={
        relation:
            5
        for relation
        in EXPECTED_RELATIONS
    }
)


paired = paired_comparison(
    base=
        base,

    adapted=
        adapted,
)

assert (
    paired[
        "accuracy_delta"
    ]
    >
    0
)

assert (
    paired[
        "macro_accuracy_delta"
    ]
    >
    0
)


gates = evaluate_acceptance_gates(
    base=
        base,

    adapted=
        adapted,

    paired=
        paired,
)

assert (
    gates[
        "all_passed"
    ]
    is True
)

assert (
    gates[
        "greenhouse_opened"
    ]
    is False
)

assert (
    gates[
        "greenhouse_authorized_by_runner"
    ]
    is False
)

assert (
    gates[
        "greenhouse_may_be_considered_after_airport_evidence_commit"
    ]
    is True
)


# 4/6 rounds to 0.666667 and must pass
# the frozen uncertain threshold.
threshold_adapted = model_result(
    relation_correct_counts={
        "same_metric_different_state":
            5,

        "same_process_different_stage":
            5,

        "related_distinct_metric":
            5,

        "unrelated":
            5,

        "uncertain":
            4,
    }
)

threshold_base = model_result(
    relation_correct_counts={
        "same_metric_different_state":
            4,

        "same_process_different_stage":
            4,

        "related_distinct_metric":
            4,

        "unrelated":
            4,

        "uncertain":
            4,
    }
)

threshold_pair = paired_comparison(
    base=
        threshold_base,

    adapted=
        threshold_adapted,
)

threshold_gates = evaluate_acceptance_gates(
    base=
        threshold_base,

    adapted=
        threshold_adapted,

    paired=
        threshold_pair,
)

assert (
    threshold_adapted[
        "uncertain_accuracy"
    ]
    ==
    0.666667
)

assert (
    threshold_gates[
        "absolute"
    ][
        "adapted_uncertain_accuracy_minimum"
    ]
    is True
)


invalid_json_adapted = model_result(
    relation_correct_counts={
        relation:
            5
        for relation
        in EXPECTED_RELATIONS
    },

    valid_count=
        29,
)

invalid_pair = paired_comparison(
    base=
        base,

    adapted=
        invalid_json_adapted,
)

invalid_gates = evaluate_acceptance_gates(
    base=
        base,

    adapted=
        invalid_json_adapted,

    paired=
        invalid_pair,
)

assert (
    invalid_gates[
        "absolute"
    ][
        "adapted_strict_json_validity_rate"
    ]
    is False
)

assert (
    invalid_gates[
        "all_passed"
    ]
    is False
)

assert (
    invalid_gates[
        "failure_action"
    ]
    ==
    "stop_v0.4_before_greenhouse"
)


low_relation = model_result(
    relation_correct_counts={
        "same_metric_different_state":
            5,

        "same_process_different_stage":
            5,

        "related_distinct_metric":
            5,

        "unrelated":
            2,

        "uncertain":
            5,
    }
)

low_relation_pair = paired_comparison(
    base=
        base,

    adapted=
        low_relation,
)

low_relation_gates = evaluate_acceptance_gates(
    base=
        base,

    adapted=
        low_relation,

    paired=
        low_relation_pair,
)

assert (
    low_relation_gates[
        "absolute"
    ][
        "adapted_per_relation_accuracy_minimum"
    ]
    is False
)

assert (
    low_relation_gates[
        "all_passed"
    ]
    is False
)


worse_adapted = model_result(
    relation_correct_counts={
        relation:
            3
        for relation
        in EXPECTED_RELATIONS
    }
)

worse_pair = paired_comparison(
    base=
        base,

    adapted=
        worse_adapted,
)

worse_gates = evaluate_acceptance_gates(
    base=
        base,

    adapted=
        worse_adapted,

    paired=
        worse_pair,
)

assert (
    worse_pair[
        "accuracy_delta"
    ]
    <
    0
)

assert (
    worse_pair[
        "macro_accuracy_delta"
    ]
    <
    0
)

assert (
    worse_gates[
        "non_regression"
    ][
        "accuracy_delta_minimum"
    ]
    is False
)

assert (
    worse_gates[
        "non_regression"
    ][
        "macro_accuracy_delta_minimum"
    ]
    is False
)

assert (
    worse_gates[
        "all_passed"
    ]
    is False
)


print(
    "[PASS] five-class metrics and conjunctive Airport gates"
)


# ============================================================
# STATIC SAFETY
# ============================================================


runner_path = (
    Path(__file__)
    .resolve()
    .parent
    / "app"
    / "adaptation"
    / "airport_evaluation_runner_v0_4_v0_2.py"
)

source = runner_path.read_text(
    encoding=
        "utf-8-sig"
)

tree = ast.parse(
    source
)


top_level_imports = []

for node in tree.body:
    if isinstance(
        node,
        ast.Import,
    ):
        top_level_imports.extend(
            alias.name
            for alias
            in node.names
        )

    elif isinstance(
        node,
        ast.ImportFrom,
    ):
        top_level_imports.append(
            node.module
            or
            ""
        )


for forbidden in (
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
):
    assert not any(
        (
            imported
            ==
            forbidden
            or
            imported.startswith(
                forbidden
                +
                "."
            )
        )
        for imported
        in top_level_imports
    )


assert ".generate(" in source

assert "_score_case" not in source

assert "candidate_continuation" not in source

assert "load_frozen_reasoning_cases" not in source

assert "do_sample=False" in source

assert "num_beams=1" in source

assert (
    "max_new_tokens="
    "\n                MAX_NEW_TOKENS"
    in source
    or
    "max_new_tokens=MAX_NEW_TOKENS"
    in source
)

assert "skip_special_tokens=False" in source

assert "build_user_message" in source

assert "CONSUMPTION_MARKER_PATH" in source


functions = {
    node.name:
        node
    for node
    in tree.body
    if isinstance(
        node,
        ast.FunctionDef,
    )
}


assert "_load_official_holdout" in functions

assert "preflight_runtime" in functions

assert "execute_evaluation" in functions


preflight_source = (
    ast.get_source_segment(
        source,
        functions[
            "preflight_runtime"
        ],
    )
    or
    ""
)

execute_source = (
    ast.get_source_segment(
        source,
        functions[
            "execute_evaluation"
        ],
    )
    or
    ""
)


assert (
    "_load_official_holdout"
    not in
    preflight_source
)

assert (
    "AIRPORT_CASES_PATH"
    not in
    preflight_source
)


marker_position = (
    execute_source.find(
        "_write_consumption_marker"
    )
)

holdout_position = (
    execute_source.find(
        "_load_official_holdout"
    )
)

assert marker_position >= 0

assert (
    holdout_position
    >
    marker_position
)


# The protected case path may only be consumed by
# _load_official_holdout(). The module-level declaration itself
# is allowed and does not read bytes.
case_path_functions = set()

for node in ast.walk(
    tree
):
    if (
        not isinstance(
            node,
            ast.Name,
        )
        or
        node.id
        !=
        "AIRPORT_CASES_PATH"
    ):
        continue

    owner = "<module>"

    for candidate in ast.walk(
        tree
    ):
        if not isinstance(
            candidate,
            ast.FunctionDef,
        ):
            continue

        end = (
            candidate.end_lineno
            or
            candidate.lineno
        )

        if (
            candidate.lineno
            <=
            node.lineno
            <=
            end
        ):
            owner = candidate.name
            break

    case_path_functions.add(
        owner
    )


assert (
    case_path_functions
    <=
    {
        "<module>",
        "_load_official_holdout",
    }
)


print(
    "[PASS] heavy imports deferred and Airport consumption isolated"
)

print(
    "[PASS] single-use marker precedes protected holdout read"
)

print(
    "[PASS] Hotel teacher-forced scorer absent"
)

print(
    "[PASS] Greenhouse cannot be opened by runner"
)

# ============================================================
# V0.2 CHAT-TEMPLATE COMPATIBILITY DELTA
# ============================================================


import copy

from transformers import AutoTokenizer

from app.adaptation.airport_evaluation_runner_v0_4_v0_2 import (
    AIRPORT_EVALUATION_RUNNER_RULE_VERSION,
    BASE_MODEL_REPOSITORY,
    BASE_MODEL_REVISION,
    _build_prompt_record,
    _synthetic_case,
)

from app.adaptation.training_dataset_canonicalizer_v0_4 import (
    build_user_message,
)


assert (
    AIRPORT_EVALUATION_RUNNER_RULE_VERSION
    ==
    "qlora_v0.4_airport_evaluation_runner_v0.2"
)


v01_runner_path = (
    Path(__file__)
    .resolve()
    .parent
    / "app"
    / "adaptation"
    / "airport_evaluation_runner_v0_4.py"
)

v02_runner_path = (
    Path(__file__)
    .resolve()
    .parent
    / "app"
    / "adaptation"
    / "airport_evaluation_runner_v0_4_v0_2.py"
)


v01_source = v01_runner_path.read_text(
    encoding="utf-8-sig"
)

v02_source = v02_runner_path.read_text(
    encoding="utf-8-sig"
)


v01_tree = ast.parse(
    v01_source
)

v02_tree = ast.parse(
    v02_source
)


def find_function(
    tree: ast.Module,
    name: str,
) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and
            node.name == name
        )
    ]

    assert len(
        matches
    ) == 1

    return matches[
        0
    ]


def find_apply_chat_template_call(
    function: ast.FunctionDef,
) -> ast.Call:
    matches = []

    for node in ast.walk(
        function
    ):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        func = node.func

        if (
            isinstance(
                func,
                ast.Attribute,
            )
            and
            func.attr
            ==
            "apply_chat_template"
        ):
            matches.append(
                node
            )

    assert len(
        matches
    ) == 1

    return matches[
        0
    ]


v01_generate = find_function(
    v01_tree,
    "_generate_case",
)

v02_generate = find_function(
    v02_tree,
    "_generate_case",
)


v01_call = find_apply_chat_template_call(
    v01_generate
)

v02_call = find_apply_chat_template_call(
    v02_generate
)


v01_return_dict = [
    keyword
    for keyword in v01_call.keywords
    if keyword.arg
    ==
    "return_dict"
]

v02_return_dict = [
    keyword
    for keyword in v02_call.keywords
    if keyword.arg
    ==
    "return_dict"
]


assert v01_return_dict == []

assert len(
    v02_return_dict
) == 1

assert isinstance(
    v02_return_dict[
        0
    ].value,
    ast.Constant,
)

assert (
    v02_return_dict[
        0
    ].value.value
    is False
)


# Remove the one approved keyword from an AST copy.
# The two _generate_case() functions must then be identical.
normalized_v02 = copy.deepcopy(
    v02_generate
)

normalized_call = (
    find_apply_chat_template_call(
        normalized_v02
    )
)

normalized_call.keywords = [
    keyword
    for keyword in normalized_call.keywords
    if keyword.arg
    !=
    "return_dict"
]


assert (
    ast.dump(
        v01_generate,
        include_attributes=False,
    )
    ==
    ast.dump(
        normalized_v02,
        include_attributes=False,
    )
)


print(
    "[PASS] _generate_case sole functional delta is return_dict=False"
)


# ============================================================
# REAL PINNED TOKENIZER ? SYNTHETIC INPUT ONLY
# ============================================================


tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_REPOSITORY,
    revision=
        BASE_MODEL_REVISION,
    trust_remote_code=
        False,
    local_files_only=
        True,
)


synthetic_case = _synthetic_case()

synthetic_record = _build_prompt_record(
    domain=
        "airport_ground_operations_synthetic_preflight",

    case=
        synthetic_case,
)

synthetic_message = build_user_message(
    synthetic_record
)

messages = [
    {
        "role":
            "user",

        "content":
            synthetic_message,
    }
]


default_result = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    truncation=False,
    return_tensors="pt",
)

fixed_result = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    truncation=False,
    return_tensors="pt",
    return_dict=False,
)


assert hasattr(
    default_result,
    "keys",
)

assert set(
    default_result.keys()
) == {
    "input_ids",
    "attention_mask",
}

assert (
    tuple(
        default_result[
            "input_ids"
        ].shape
    )
    ==
    (
        1,
        174,
    )
)

assert (
    getattr(
        fixed_result,
        "ndim",
        None,
    )
    ==
    2
)

assert (
    tuple(
        fixed_result.shape
    )
    ==
    (
        1,
        174,
    )
)

assert (
    default_result[
        "input_ids"
    ].tolist()
    ==
    fixed_result.tolist()
)


print(
    "[PASS] pinned tokenizer default BatchEncoding reproduced"
)

print(
    "[PASS] return_dict=False produces Tensor [1,174]"
)

print(
    "[PASS] compatibility delta preserves exact input token IDs"
)

print(
    "[PASS] compatibility test uses synthetic prompt only"
)

print()

print(
    "DATALENS QLORA v0.4 AIRPORT EVALUATION RUNNER TEST v0.2: PASS"
)
