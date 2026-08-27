from __future__ import annotations

import ast

from pathlib import Path


TEST_RULE_VERSION = (
    "report_pdf_error_privacy_test_v0.1"
)

SAFE_ERROR_CODE = (
    "server_owned_pdf_generation_failed"
)

SAFE_ERROR_MESSAGE = (
    "La génération locale du PDF "
    "server-owned a échoué."
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

TARGET = (
    ROOT
    / "app"
    / "api"
    / "report_selection.py"
)


def parse_target() -> ast.Module:

    return ast.parse(
        TARGET.read_text(
            encoding="utf-8-sig"
        ),
        filename=(
            "app/api/report_selection.py"
        ),
    )


def find_export_function(
) -> ast.FunctionDef:

    matches = [
        node
        for node
        in parse_target().body
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and
            node.name
            ==
            "export_selected_report_pdf"
        )
    ]

    if len(
        matches
    ) != 1:
        raise AssertionError(
            (
                "export_selected_report_pdf "
                "not found exactly once."
            )
        )

    return matches[
        0
    ]


def test_rule_version() -> None:

    if (
        TEST_RULE_VERSION
        !=
        "report_pdf_error_privacy_test_v0.1"
    ):
        raise AssertionError(
            "Unexpected rule version."
        )


def test_public_500_does_not_reference_exception(
) -> None:

    function = (
        find_export_function()
    )

    handlers = [
        node
        for node
        in ast.walk(
            function
        )
        if (
            isinstance(
                node,
                ast.ExceptHandler,
            )
            and
            isinstance(
                node.type,
                ast.Name,
            )
            and
            node.type.id
            ==
            "Exception"
        )
    ]

    if len(
        handlers
    ) != 1:
        raise AssertionError(
            (
                "Expected one generic "
                "PDF exception handler."
            )
        )

    handler = (
        handlers[
            0
        ]
    )

    http_calls = [
        node
        for node
        in ast.walk(
            handler
        )
        if (
            isinstance(
                node,
                ast.Call,
            )
            and
            isinstance(
                node.func,
                ast.Name,
            )
            and
            node.func.id
            ==
            "HTTPException"
        )
    ]

    if len(
        http_calls
    ) != 1:
        raise AssertionError(
            (
                "Expected one HTTPException "
                "inside PDF failure handler."
            )
        )

    detail = None

    for keyword in (
        http_calls[
            0
        ]
        .keywords
    ):
        if (
            keyword.arg
            ==
            "detail"
        ):
            detail = (
                keyword.value
            )

    if detail is None:
        raise AssertionError(
            "PDF HTTPException has no detail."
        )

    if any(
        (
            isinstance(
                node,
                ast.Name,
            )
            and
            node.id
            ==
            "error"
        )
        for node
        in ast.walk(
            detail
        )
    ):
        raise AssertionError(
            (
                "Public PDF detail references "
                "the internal exception."
            )
        )


def test_safe_error_contract_present(
) -> None:

    function = (
        find_export_function()
    )

    http_calls = [
        node
        for node
        in ast.walk(
            function
        )
        if (
            isinstance(
                node,
                ast.Call,
            )
            and
            isinstance(
                node.func,
                ast.Name,
            )
            and
            node.func.id
            ==
            "HTTPException"
        )
    ]

    error_details = []

    for call in http_calls:

        status_code = None
        detail = None

        for keyword in (
            call.keywords
        ):
            if (
                keyword.arg
                ==
                "status_code"
            ):
                status_code = (
                    keyword.value
                )

            if (
                keyword.arg
                ==
                "detail"
            ):
                detail = (
                    keyword.value
                )

        if not (
            isinstance(
                status_code,
                ast.Constant,
            )
            and
            status_code.value
            ==
            500
        ):
            continue

        if isinstance(
            detail,
            ast.Dict,
        ):
            error_details.append(
                detail
            )

    if len(
        error_details
    ) != 1:
        raise AssertionError(
            (
                "Expected exactly one structured "
                "PDF HTTP 500 detail."
            )
        )

    detail = (
        error_details[
            0
        ]
    )

    payload = {}

    for key, value in zip(
        detail.keys,
        detail.values,
    ):

        if not (
            isinstance(
                key,
                ast.Constant,
            )
            and
            isinstance(
                key.value,
                str,
            )
        ):
            continue

        if (
            isinstance(
                value,
                ast.Constant,
            )
            and
            isinstance(
                value.value,
                str,
            )
        ):
            payload[
                key.value
            ] = (
                value.value
            )

    if (
        payload.get(
            "error"
        )
        !=
        SAFE_ERROR_CODE
    ):
        raise AssertionError(
            "Safe PDF error code missing."
        )

    if (
        payload.get(
            "message"
        )
        !=
        SAFE_ERROR_MESSAGE
    ):
        raise AssertionError(
            (
                "Unexpected safe PDF "
                "error message."
            )
        )


def main() -> None:

    print(
        "=== DATALENS REPORT PDF "
        "ERROR PRIVACY v0.1 ==="
    )
    print()

    tests = [
        (
            "Safe error rule version",
            test_rule_version,
        ),
        (
            "Public PDF 500 suppresses exception",
            test_public_500_does_not_reference_exception,
        ),
        (
            "Safe PDF error contract present",
            test_safe_error_contract_present,
        ),
    ]

    for label, test in tests:
        test()

        print(
            f"[PASS] {label}"
        )

    print()
    print(
        "PASS - 3/3 Report PDF "
        "error privacy checks"
    )
    print(
        f"Rule: {TEST_RULE_VERSION}"
    )


if __name__ == "__main__":
    main()
