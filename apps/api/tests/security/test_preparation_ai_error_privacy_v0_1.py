from __future__ import annotations

import ast

from pathlib import Path


TEST_RULE_VERSION = (
    "preparation_ai_error_privacy_test_v0.1"
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


MODEL_FILES = (
    "app/preparation/analysis_output_explanation.py",
    "app/preparation/dataset_identity_explanation.py",
    "app/preparation/semantic_review.py",
)


def parse(
    relative: str,
):
    path = (
        ROOT
        / relative
    )

    return ast.parse(
        path.read_text(
            encoding="utf-8-sig"
        ),
        filename=relative,
    )


def exception_type_names(
    handler: ast.ExceptHandler,
) -> set[str]:

    node = handler.type

    if isinstance(
        node,
        ast.Name,
    ):
        return {
            node.id
        }

    if isinstance(
        node,
        ast.Tuple,
    ):
        return {
            item.id
            for item
            in node.elts
            if isinstance(
                item,
                ast.Name,
            )
        }

    return set()


def test_rule_version() -> None:

    if (
        TEST_RULE_VERSION
        !=
        "preparation_ai_error_privacy_test_v0.1"
    ):
        raise AssertionError(
            "Unexpected test rule version."
        )


def test_http_error_body_is_never_read() -> None:

    for relative in MODEL_FILES:

        tree = parse(
            relative
        )

        handlers = [
            node
            for node
            in ast.walk(
                tree
            )
            if (
                isinstance(
                    node,
                    ast.ExceptHandler,
                )
                and
                "HTTPError"
                in exception_type_names(
                    node
                )
            )
        ]

        if len(
            handlers
        ) != 1:
            raise AssertionError(
                (
                    f"{relative}: expected exactly "
                    f"one HTTPError handler, "
                    f"found {len(handlers)}."
                )
            )

        handler = (
            handlers[0]
        )

        exception_name = (
            handler.name
        )

        for node in ast.walk(
            handler
        ):

            if not isinstance(
                node,
                ast.Call,
            ):
                continue

            if not isinstance(
                node.func,
                ast.Attribute,
            ):
                continue

            if (
                node.func.attr
                !=
                "read"
            ):
                continue

            if (
                isinstance(
                    node.func.value,
                    ast.Name,
                )
                and
                node.func.value.id
                ==
                exception_name
            ):
                raise AssertionError(
                    (
                        f"{relative}: HTTP error "
                        "response body is read."
                    )
                )


def test_ai_error_fields_are_sanitized() -> None:

    targets = {
        (
            "app/api/"
            "preparation_output_explanation.py"
        ):
            "explain_preparation_analysis_output",

        (
            "app/api/"
            "preparation_identity.py"
        ):
            "inspect_preparation_identity",
    }

    for (
        relative,
        function_name,
    ) in targets.items():

        tree = parse(
            relative
        )

        functions = [
            node
            for node
            in tree.body
            if (
                isinstance(
                    node,
                    ast.FunctionDef,
                )
                and
                node.name
                ==
                function_name
            )
        ]

        if len(
            functions
        ) != 1:
            raise AssertionError(
                (
                    f"{relative}: expected function "
                    f"{function_name}."
                )
            )

        for node in ast.walk(
            functions[0]
        ):

            if not isinstance(
                node,
                ast.Assign,
            ):
                continue

            targets_ai_error = any(
                isinstance(
                    target,
                    ast.Name,
                )
                and
                target.id
                ==
                "ai_error"

                for target
                in node.targets
            )

            if not targets_ai_error:
                continue

            if (
                isinstance(
                    node.value,
                    ast.Call,
                )
                and
                isinstance(
                    node.value.func,
                    ast.Name,
                )
                and
                node.value.func.id
                in {
                    "str",
                    "repr",
                }
            ):
                raise AssertionError(
                    (
                        f"{relative}: ai_error "
                        "renders exception text."
                    )
                )


def test_semantic_report_does_not_render_model_error(
) -> None:

    tree = parse(
        "app/preparation/semantic_review.py"
    )

    functions = [
        node
        for node
        in tree.body
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and
            node.name
            ==
            "review_quality_semantics"
        )
    ]

    if len(
        functions
    ) != 1:
        raise AssertionError(
            "review_quality_semantics not found."
        )

    for handler in [
        node
        for node
        in ast.walk(
            functions[0]
        )
        if isinstance(
            node,
            ast.ExceptHandler,
        )
    ]:

        types = (
            exception_type_names(
                handler
            )
        )

        if not (
            {
                "RuntimeError",
                "ValueError",
            }
            <=
            types
        ):
            continue

        if (
            handler.name
            is not None
        ):
            raise AssertionError(
                (
                    "Semantic model failure keeps "
                    "a public exception variable."
                )
            )


def main() -> None:

    print(
        "=== DATALENS PREPARATION AI "
        "ERROR PRIVACY v0.1 ==="
    )
    print()

    tests = [
        (
            "Safe error rule version",
            test_rule_version,
        ),
        (
            "HTTP error bodies are not read",
            test_http_error_body_is_never_read,
        ),
        (
            "AI error fields are sanitized",
            test_ai_error_fields_are_sanitized,
        ),
        (
            "Semantic failure details suppressed",
            test_semantic_report_does_not_render_model_error,
        ),
    ]

    for label, test in tests:
        test()

        print(
            f"[PASS] {label}"
        )

    print()
    print(
        "PASS - 4/4 Preparation AI "
        "error privacy checks"
    )
    print(
        f"Rule: {TEST_RULE_VERSION}"
    )


if __name__ == "__main__":
    main()
