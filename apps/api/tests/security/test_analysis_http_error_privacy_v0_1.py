from __future__ import annotations

import ast

from pathlib import Path


TEST_RULE_VERSION = (
    "analysis_http_error_privacy_test_v0.1"
)

SAFE_AI_DETAIL = (
    "Local AI processing is unavailable "
    "or returned an invalid response."
)

SAFE_CONTEXTUALIZED_PERSISTENCE_422_MESSAGE = (
    "Contextualized analysis artifact "
    "persistence request is invalid."
)

SAFE_CONTEXTUALIZED_PERSISTENCE_503_MESSAGE = (
    "Contextualized analysis artifacts "
    "could not be persisted."
)

SAFE_PDF_DETAIL = (
    "La génération locale du PDF a échoué."
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

ANALYSIS_RUN = (
    ROOT
    / "app"
    / "api"
    / "analysis_run.py"
)


def source_text() -> str:
    return ANALYSIS_RUN.read_text(
        encoding="utf-8-sig"
    )


def tree() -> ast.Module:
    return ast.parse(
        source_text(),
        filename="app/api/analysis_run.py",
    )


def exception_names(
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
            for item in node.elts
            if isinstance(
                item,
                ast.Name,
            )
        }

    return set()


def http_exception_calls(
    node: ast.AST,
):
    result = []

    for item in ast.walk(
        node
    ):
        if not isinstance(
            item,
            ast.Call,
        ):
            continue

        if (
            isinstance(
                item.func,
                ast.Name,
            )
            and
            item.func.id
            ==
            "HTTPException"
        ):
            result.append(
                item
            )

    return result


def keyword_value(
    call: ast.Call,
    name: str,
):
    for keyword in call.keywords:
        if (
            keyword.arg
            ==
            name
        ):
            return keyword.value

    return None


def literal_int(
    node: ast.AST | None,
):
    if (
        isinstance(
            node,
            ast.Constant,
        )
        and
        isinstance(
            node.value,
            int,
        )
    ):
        return node.value

    return None


def literal_string(
    node: ast.AST | None,
):
    if (
        isinstance(
            node,
            ast.Constant,
        )
        and
        isinstance(
            node.value,
            str,
        )
    ):
        return node.value

    return None


def literal_dict_string(
    node: ast.AST | None,
    key: str,
):
    if not isinstance(
        node,
        ast.Dict,
    ):
        return None

    for (
        key_node,
        value_node,
    ) in zip(
        node.keys,
        node.values,
    ):
        if (
            literal_string(
                key_node
            )
            ==
            key
        ):
            return literal_string(
                value_node
            )

    return None


def test_rule_version() -> None:
    if (
        TEST_RULE_VERSION
        !=
        "analysis_http_error_privacy_test_v0.1"
    ):
        raise AssertionError(
            "Unexpected rule version."
        )


def test_runtime_503_details_are_static() -> None:
    handlers = []

    for node in ast.walk(
        tree()
    ):
        if not isinstance(
            node,
            ast.ExceptHandler,
        ):
            continue

        if (
            "RuntimeError"
            not in
            exception_names(
                node
            )
        ):
            continue

        calls = [
            call
            for call
            in http_exception_calls(
                node
            )
            if (
                literal_int(
                    keyword_value(
                        call,
                        "status_code",
                    )
                )
                ==
                503
            )
        ]

        if calls:
            handlers.append(
                (
                    node,
                    calls,
                )
            )

    if len(
        handlers
    ) != 6:
        raise AssertionError(
            (
                "Expected exactly 6 RuntimeError "
                "HTTP 503 handlers; found "
                f"{len(handlers)}."
            )
        )

    generic_count = 0
    contextualized_persistence_count = 0

    for _, calls in handlers:
        if len(
            calls
        ) != 1:
            raise AssertionError(
                "Unexpected RuntimeError HTTP structure."
            )

        detail = keyword_value(
            calls[0],
            "detail",
        )

        if (
            literal_string(
                detail
            )
            ==
            SAFE_AI_DETAIL
        ):
            generic_count += 1
            continue

        error_code = (
            literal_dict_string(
                detail,
                "error",
            )
        )

        message = (
            literal_dict_string(
                detail,
                "message",
            )
        )

        if (
            error_code
            ==
            "contextualized_artifact_persistence_failed"
            and
            message
            ==
            SAFE_CONTEXTUALIZED_PERSISTENCE_503_MESSAGE
        ):
            contextualized_persistence_count += 1
            continue

        raise AssertionError(
            (
                "RuntimeError HTTP 503 exposes "
                "a non-static technical detail."
            )
        )

    if generic_count != 5:
        raise AssertionError(
            (
                "Expected exactly 5 generic "
                "static RuntimeError HTTP 503 "
                f"handlers; found {generic_count}."
            )
        )

    if (
        contextualized_persistence_count
        !=
        1
    ):
        raise AssertionError(
            (
                "Expected exactly one static "
                "contextualized persistence "
                "RuntimeError HTTP 503 handler."
            )
        )


def test_contextualized_persistence_422_detail_is_static(
) -> None:

    matches = []

    for node in ast.walk(
        tree()
    ):
        if not isinstance(
            node,
            ast.ExceptHandler,
        ):
            continue

        if (
            "ValueError"
            not in
            exception_names(
                node
            )
        ):
            continue

        calls = [
            call
            for call
            in http_exception_calls(
                node
            )
            if (
                literal_int(
                    keyword_value(
                        call,
                        "status_code",
                    )
                )
                ==
                422
            )
        ]

        for call in calls:
            detail = keyword_value(
                call,
                "detail",
            )

            if (
                literal_dict_string(
                    detail,
                    "error",
                )
                ==
                "invalid_contextualized_artifact_persistence"
            ):
                matches.append(
                    detail
                )

    if len(
        matches
    ) != 1:
        raise AssertionError(
            (
                "Expected exactly one "
                "contextualized persistence "
                "ValueError HTTP 422 handler; "
                f"found {len(matches)}."
            )
        )

    message = (
        literal_dict_string(
            matches[0],
            "message",
        )
    )

    if (
        message
        !=
        SAFE_CONTEXTUALIZED_PERSISTENCE_422_MESSAGE
    ):
        raise AssertionError(
            (
                "Contextualized persistence HTTP 422 "
                "exposes a non-static technical detail."
            )
        )


def test_pdf_error_is_static() -> None:
    functions = [
        node
        for node
        in tree().body
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and
            node.name
            ==
            "export_analysis_pdf"
        )
    ]

    if len(
        functions
    ) != 1:
        raise AssertionError(
            "export_analysis_pdf not found."
        )

    calls = [
        call
        for call
        in http_exception_calls(
            functions[0]
        )
        if (
            literal_int(
                keyword_value(
                    call,
                    "status_code",
                )
            )
            ==
            500
        )
    ]

    if len(
        calls
    ) != 1:
        raise AssertionError(
            "Expected one PDF HTTP 500 handler."
        )

    detail = keyword_value(
        calls[0],
        "detail",
    )

    if (
        literal_string(
            detail
        )
        !=
        SAFE_PDF_DETAIL
    ):
        raise AssertionError(
            "PDF error exposes internal exception details."
        )


def test_trace_write_error_is_not_public() -> None:
    if (
        "trace_write.error"
        in
        source_text()
    ):
        raise AssertionError(
            (
                "Trace persistence error text "
                "is referenced by HTTP/report code."
            )
        )


def test_view_builder_exception_is_not_rendered(
) -> None:

    functions = [
        node
        for node
        in tree().body
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and
            node.name
            ==
            "prepare_analysis_datasets"
        )
    ]

    if len(
        functions
    ) != 1:
        raise AssertionError(
            "prepare_analysis_datasets not found."
        )

    handlers = [
        node
        for node
        in ast.walk(
            functions[0]
        )
        if (
            isinstance(
                node,
                ast.ExceptHandler,
            )
            and
            "Exception"
            in
            exception_names(
                node
            )
        )
    ]

    if len(
        handlers
    ) != 1:
        raise AssertionError(
            "Expected one view-builder Exception handler."
        )

    if (
        handlers[0].name
        is not None
    ):
        raise AssertionError(
            (
                "View-builder exception is still "
                "bound for possible rendering."
            )
        )

    if (
        "Analytical View Builder error"
        in
        source_text()
    ):
        raise AssertionError(
            "View-builder internal error text remains."
        )


def main() -> None:
    print(
        "=== DATALENS ANALYSIS HTTP "
        "ERROR PRIVACY v0.1 ==="
    )
    print()

    tests = [
        (
            "Safe error rule version",
            test_rule_version,
        ),
        (
            "Runtime 503 details are static",
            test_runtime_503_details_are_static,
        ),
        (
            "Contextualized persistence 422 is static",
            test_contextualized_persistence_422_detail_is_static,
        ),
        (
            "PDF error detail is static",
            test_pdf_error_is_static,
        ),
        (
            "Trace write error is not public",
            test_trace_write_error_is_not_public,
        ),
        (
            "View-builder details suppressed",
            test_view_builder_exception_is_not_rendered,
        ),
    ]

    for label, test in tests:
        test()
        print(
            f"[PASS] {label}"
        )

    print()
    print(
        "PASS - 6/6 Analysis HTTP "
        "error privacy checks"
    )
    print(
        f"Rule: {TEST_RULE_VERSION}"
    )


if __name__ == "__main__":
    main()
