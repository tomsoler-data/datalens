from __future__ import annotations


import ast
import importlib
import inspect


from pathlib import Path


RUNNER_MODULE_NAME = (
    "app.adaptation."
    "hospital_independent_evaluation_runner_v0_4_v0_1"
)


RUNNER_PATH = Path(
    "app/adaptation/"
    "hospital_independent_evaluation_runner_v0_4_v0_1.py"
)


EXPECTED_VISIBLE_FIELDS = {
    "domain",
    "left_metric",
    "left_description",
    "right_metric",
    "right_description",
}


FORBIDDEN_GOLD_FIELDS = {
    "gold_relation",
    "gold_reason",
}


HEAVY_MODULE_PREFIXES = (
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
)


HISTORICAL_EXECUTION_TOKENS = (
    "airport_ground_operations",
    "hotel_operations",
    "commercial_greenhouse_operations",
)


def load_runner():

    return importlib.import_module(
        RUNNER_MODULE_NAME
    )


def load_tree() -> ast.Module:

    return ast.parse(
        RUNNER_PATH.read_text(
            encoding="utf-8"
        ),
        filename=str(
            RUNNER_PATH
        ),
    )


def function_node(
    tree: ast.Module,
    name: str,
) -> ast.FunctionDef:

    matches = [
        node

        for node
        in tree.body

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


def test_runner_import_is_lightweight() -> None:

    tree = load_tree()


    imported_modules = []


    for node in tree.body:

        if isinstance(
            node,
            ast.Import,
        ):

            imported_modules.extend(
                alias.name

                for alias
                in node.names
            )


        elif isinstance(
            node,
            ast.ImportFrom,
        ):

            if node.module:

                imported_modules.append(
                    node.module
                )


    heavy = [
        module

        for module
        in imported_modules

        if any(
            module.startswith(
                prefix
            )

            for prefix
            in HEAVY_MODULE_PREFIXES
        )
    ]


    assert heavy == []


def test_runner_import_does_not_open_holdout() -> None:

    tree = load_tree()


    top_level_calls = [
        node

        for node
        in tree.body

        if isinstance(
            node,
            ast.Expr,
        )
        and
        isinstance(
            node.value,
            ast.Call,
        )
    ]


    assert top_level_calls == []


def test_runner_has_no_historical_execution_authority() -> None:

    source = RUNNER_PATH.read_text(
        encoding="utf-8"
    ).casefold()


    for token in HISTORICAL_EXECUTION_TOKENS:

        assert token not in source


def test_label_blind_record_has_exact_visible_keys() -> None:

    runner = load_runner()


    case = {
        "case_id":
            "synthetic-001",

        "domain":
            "hospital_emergency_department_operations",

        "left_metric":
            "synthetic_left_metric",

        "left_description":
            "Synthetic left metric definition.",

        "right_metric":
            "synthetic_right_metric",

        "right_description":
            "Synthetic right metric definition.",

        "gold_relation":
            "__SECRET_RELATION_SENTINEL__",

        "gold_reason":
            "__SECRET_REASON_SENTINEL__",
    }


    visible = runner.build_label_blind_record(
        case
    )


    assert set(
        visible
    ) == EXPECTED_VISIBLE_FIELDS


    assert (
        set(
            visible
        )
        &
        FORBIDDEN_GOLD_FIELDS
    ) == set()


def test_prompt_does_not_receive_gold_values() -> None:

    runner = load_runner()


    relation_sentinel = (
        "__SECRET_RELATION_SENTINEL_7C1A__"
    )

    reason_sentinel = (
        "__SECRET_REASON_SENTINEL_91FE__"
    )


    case = {
        "case_id":
            "synthetic-002",

        "domain":
            "hospital_emergency_department_operations",

        "left_metric":
            "synthetic_arrivals",

        "left_description":
            "Synthetic definition for arrivals.",

        "right_metric":
            "synthetic_completions",

        "right_description":
            "Synthetic definition for completions.",

        "gold_relation":
            relation_sentinel,

        "gold_reason":
            reason_sentinel,
    }


    prompt = runner.build_label_blind_prompt(
        case
    )


    assert relation_sentinel not in prompt

    assert reason_sentinel not in prompt


    assert (
        "synthetic_arrivals"
        in
        prompt
    )

    assert (
        "synthetic_completions"
        in
        prompt
    )


def test_execution_input_contains_no_gold() -> None:

    runner = load_runner()


    case = {
        "case_id":
            "synthetic-003",

        "domain":
            "hospital_emergency_department_operations",

        "left_metric":
            "left_metric",

        "left_description":
            "Left metric definition.",

        "right_metric":
            "right_metric",

        "right_description":
            "Right metric definition.",

        "gold_relation":
            "__SECRET_RELATION_SENTINEL__",

        "gold_reason":
            "__SECRET_REASON_SENTINEL__",
    }


    execution_input = runner.build_execution_input(
        case
    )


    assert set(
        execution_input
    ) == {
        "case_id",
        "prompt",
    }


    serialized = repr(
        execution_input
    )


    assert "__SECRET_RELATION_SENTINEL__" not in serialized

    assert "__SECRET_REASON_SENTINEL__" not in serialized


def test_prompt_functions_do_not_subscript_gold_fields() -> None:

    tree = load_tree()


    function_names = (
        "build_label_blind_record",
        "build_label_blind_prompt",
        "build_execution_input",
        "build_execution_inputs",
    )


    for name in function_names:

        node = function_node(
            tree,
            name,
        )


        forbidden_subscripts = []


        for child in ast.walk(
            node
        ):

            if not isinstance(
                child,
                ast.Subscript,
            ):

                continue


            slice_node = child.slice


            if (
                isinstance(
                    slice_node,
                    ast.Constant,
                )
                and
                slice_node.value
                in
                FORBIDDEN_GOLD_FIELDS
            ):

                forbidden_subscripts.append(
                    slice_node.value
                )


        assert forbidden_subscripts == []


def test_no_model_generation_code_exists_yet() -> None:

    tree = load_tree()


    attribute_calls = []


    for node in ast.walk(
        tree
    ):

        if (
            isinstance(
                node,
                ast.Call,
            )
            and
            isinstance(
                node.func,
                ast.Attribute,
            )
        ):

            attribute_calls.append(
                node.func.attr
            )


    assert "generate" not in attribute_calls

    assert "from_pretrained" not in attribute_calls


def test_frozen_authority_constants() -> None:

    runner = load_runner()


    assert (
        runner.EXPECTED_PROTOCOL_COMMIT
        ==
        "98e73521d633406c8de728aa667dc2525f6787a3"
    )

    assert (
        runner.EXPECTED_FROZEN_HOLDOUT_COMMIT
        ==
        "510f098a03a0e3263f607bcb68c31818b6e1c13b"
    )

    assert (
        runner.EXPECTED_HOLDOUT_CASES_SHA256
        ==
        "92e4f21e7323cd053fd5f53b51e1493297fdd28cfb717be56f7be963f298d9fc"
    )

    assert (
        runner.EXPECTED_HOLDOUT_FREEZE_SHA256
        ==
        "67baedb38d7807348d7ee6436f7f1a3730c5eeb3f1bd68bc1e2556a030411bed"
    )


def main() -> None:

    tests = [
        test_runner_import_is_lightweight,
        test_runner_import_does_not_open_holdout,
        test_runner_has_no_historical_execution_authority,
        test_label_blind_record_has_exact_visible_keys,
        test_prompt_does_not_receive_gold_values,
        test_execution_input_contains_no_gold,
        test_prompt_functions_do_not_subscript_gold_fields,
        test_no_model_generation_code_exists_yet,
        test_frozen_authority_constants,
    ]


    print(
        "=== R8 HOSPITAL LABEL-BLIND "
        "RUNNER STATIC CONTRACT v0.1 ==="
    )

    print()


    for test in tests:

        test()

        print(
            f"[PASS] {test.__name__}"
        )


    print()

    print(
        "PASS - hospital label-blind runner "
        "static contract v0.1"
    )


if __name__ == "__main__":

    main()
