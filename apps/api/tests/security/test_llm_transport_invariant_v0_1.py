from __future__ import annotations

import ast
import inspect

from pathlib import Path
from unittest.mock import patch
from urllib.request import Request

import app.security.llm_egress as llm_egress

from app.security.llm_payload import (
    LLMPayloadClass,
    LLMPayloadPrivacyError,
)


TEST_RULE_VERSION = (
    "llm_transport_invariant_test_v0.1"
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

APP_ROOT = (
    ROOT
    / "app"
)


EXCLUDED_PREFIXES = (
    "app/evals/",
    "app/evaluation/",
)


ALLOWED_DIRECT_MODEL_FILE = (
    "app/security/llm_payload.py"
)

ALLOWED_URLOPEN_FILE = (
    "app/security/llm_egress.py"
)


PREPARATION_CLASSES = {
    (
        "app/preparation/"
        "analysis_output_explanation.py"
    ):
        "DETERMINISTIC_EVIDENCE",

    (
        "app/preparation/"
        "dataset_identity_explanation.py"
    ):
        "DETERMINISTIC_EVIDENCE",

    (
        "app/preparation/"
        "semantic_review.py"
    ):
        "SEMANTIC_VALUE_SAMPLE",
}


def relative_path(
    path: Path,
) -> str:

    return (
        path
        .relative_to(ROOT)
        .as_posix()
    )


def production_python_files():

    for path in APP_ROOT.rglob(
        "*.py"
    ):

        relative = (
            relative_path(
                path
            )
        )

        if (
            ".before_"
            in path.name
        ):
            continue

        if any(
            relative.startswith(
                prefix
            )
            for prefix
            in EXCLUDED_PREFIXES
        ):
            continue

        yield (
            path
        )


def parse_file(
    path: Path,
):

    return ast.parse(
        path.read_text(
            encoding="utf-8-sig"
        ),
        filename=(
            relative_path(
                path
            )
        ),
    )


def call_name(
    node: ast.Call,
):

    if isinstance(
        node.func,
        ast.Name,
    ):
        return (
            node.func.id
        )

    if isinstance(
        node.func,
        ast.Attribute,
    ):
        return (
            node.func.attr
        )

    return None


def has_keyword(
    node: ast.Call,
    name: str,
) -> bool:

    return any(
        keyword.arg
        ==
        name

        for keyword
        in node.keywords
    )


def test_raw_rows_blocked_before_urlopen(
) -> None:

    request = Request(
        (
            "http://127.0.0.1:"
            "11434/api/chat"
        )
    )

    with patch.object(
        llm_egress._LOCAL_LLM_OPENER,
        "open",
    ) as mocked_transport:

        try:
            (
                llm_egress
                .open_local_llm_request(
                    request,
                    payload_class=(
                        LLMPayloadClass
                        .TABULAR_RAW_ROWS
                    ),
                    timeout=1.0,
                )
            )

        except LLMPayloadPrivacyError:
            pass

        else:
            raise AssertionError(
                (
                    "TABULAR_RAW_ROWS "
                    "was not rejected."
                )
            )

        if mocked_transport.called:
            raise AssertionError(
                (
                    "urllib transport was reached "
                    "before privacy rejection."
                )
            )


def test_no_direct_model_calls(
) -> None:

    violations = []

    for path in production_python_files():

        relative = (
            relative_path(
                path
            )
        )

        if (
            relative
            ==
            ALLOWED_DIRECT_MODEL_FILE
        ):
            continue

        tree = (
            parse_file(
                path
            )
        )

        for node in ast.walk(
            tree
        ):

            if not isinstance(
                node,
                ast.Call,
            ):
                continue

            if (
                call_name(node)
                in {
                    "chat",
                    "embed",
                }
            ):
                violations.append(
                    (
                        relative,
                        node.lineno,
                        call_name(node),
                    )
                )

    if violations:
        raise AssertionError(
            (
                "Direct production model "
                f"calls detected: {violations}"
            )
        )


def test_classified_calls_require_payload_class(
) -> None:

    guarded_calls = {
        "classified_llm_chat",
        "classified_llm_embed",
        "open_local_llm_request",
    }

    violations = []

    for path in production_python_files():

        tree = (
            parse_file(
                path
            )
        )

        relative = (
            relative_path(
                path
            )
        )

        for node in ast.walk(
            tree
        ):

            if not isinstance(
                node,
                ast.Call,
            ):
                continue

            name = (
                call_name(
                    node
                )
            )

            if (
                name
                not in
                guarded_calls
            ):
                continue

            if not has_keyword(
                node,
                "payload_class",
            ):
                violations.append(
                    (
                        relative,
                        node.lineno,
                        name,
                    )
                )

    if violations:
        raise AssertionError(
            (
                "Model transport without "
                f"payload_class: {violations}"
            )
        )


def test_urlopen_has_single_owner(
) -> None:

    direct_urlopen_calls = []

    opener_calls = []


    for path in production_python_files():

        tree = (
            parse_file(
                path
            )
        )

        relative = (
            relative_path(
                path
            )
        )


        for node in ast.walk(
            tree
        ):

            if not isinstance(
                node,
                ast.Call,
            ):
                continue


            name = (
                call_name(
                    node
                )
            )


            if (
                name
                ==
                "urlopen"
            ):
                direct_urlopen_calls.append(
                    (
                        relative,
                        node.lineno,
                    )
                )


            if (
                name
                ==
                "build_opener"
            ):
                opener_calls.append(
                    (
                        relative,
                        node.lineno,
                    )
                )


    if direct_urlopen_calls:
        raise AssertionError(
            (
                "Direct production urlopen "
                "calls are forbidden: "
                f"{direct_urlopen_calls}"
            )
        )


    if (
        len(
            opener_calls
        )
        !=
        1
    ):
        raise AssertionError(
            (
                "Expected exactly one production "
                "urllib opener construction; found "
                f"{opener_calls}"
            )
        )


    (
        owner,
        _line,
    ) = (
        opener_calls[
            0
        ]
    )


    if (
        owner
        !=
        ALLOWED_URLOPEN_FILE
    ):
        raise AssertionError(
            (
                "urllib opener is owned by an "
                "unauthorized module: "
                f"{owner}"
            )
        )



def test_egress_payload_class_is_required(
) -> None:

    signature = (
        inspect.signature(
            llm_egress
            .open_local_llm_request
        )
    )

    parameter = (
        signature.parameters[
            "payload_class"
        ]
    )

    if (
        parameter.kind
        !=
        inspect.Parameter.KEYWORD_ONLY
    ):
        raise AssertionError(
            (
                "payload_class must be "
                "keyword-only."
            )
        )

    if (
        parameter.default
        is not
        inspect.Parameter.empty
    ):
        raise AssertionError(
            (
                "payload_class must not "
                "have a default."
            )
        )


def test_preparation_classifications(
) -> None:

    for (
        relative,
        expected_class,
    ) in PREPARATION_CLASSES.items():

        source = (
            (ROOT / relative)
            .read_text(
                encoding="utf-8-sig"
            )
        )

        if (
            f".{expected_class}"
            not in
            source
        ):
            raise AssertionError(
                (
                    f"{relative} does not "
                    f"declare {expected_class}."
                )
            )


def main() -> None:

    print(
        "=== DATALENS LLM TRANSPORT "
        "INVARIANT v0.1 ==="
    )
    print()

    tests = [
        (
            "Raw rows blocked before urllib transport",
            test_raw_rows_blocked_before_urlopen,
        ),
        (
            "No direct production chat/embed calls",
            test_no_direct_model_calls,
        ),
        (
            "All guarded calls declare payload class",
            test_classified_calls_require_payload_class,
        ),
        (
            "urllib opener is centralized",
            test_urlopen_has_single_owner,
        ),
        (
            "urllib payload class is required",
            test_egress_payload_class_is_required,
        ),
        (
            "Preparation payload classes locked",
            test_preparation_classifications,
        ),
    ]

    for label, test in tests:
        test()
        print(
            f"[PASS] {label}"
        )

    print()
    print(
        "PASS - 6/6 LLM transport "
        "invariant checks"
    )
    print(
        f"Rule: {TEST_RULE_VERSION}"
    )


if __name__ == "__main__":
    main()
