from __future__ import annotations


from http.server import (
    BaseHTTPRequestHandler,
    HTTPServer,
)

from pathlib import (
    Path,
)

from threading import (
    Thread,
)

from urllib.request import (
    ProxyHandler,
    Request,
)


import app.ai.provider as provider_module

import app.security.llm_egress as llm_egress_module

from app.security.llm_payload import (
    LLMPayloadClass,
)


TEST_RULE_VERSION = (
    "llm_redirect_egress_test_v0.1"
)


class QuietHandler(
    BaseHTTPRequestHandler
):
    def log_message(
        self,
        format,
        *args,
    ) -> None:

        del (
            format,
            args,
        )


def test_rule_version(
) -> None:

    assert (
        TEST_RULE_VERSION
        ==
        "llm_redirect_egress_test_v0.1"
    )


def test_urllib_proxy_environment_is_disabled(
) -> None:

    # ProxyHandler({}) deliberately removes urllib's
    # environment-derived default proxy handler.
    #
    # An empty ProxyHandler may not remain visible in
    # OpenerDirector.handlers because it exposes no active
    # proxy protocol methods. Therefore runtime absence of
    # ProxyHandler is the expected state.

    proxy_handlers = [
        handler

        for handler
        in (
            llm_egress_module
            ._LOCAL_LLM_OPENER
            .handlers
        )

        if isinstance(
            handler,
            ProxyHandler,
        )
    ]


    assert (
        proxy_handlers
        ==
        []
    )


    source = (
        Path(
            llm_egress_module.__file__
        )
        .read_text(
            encoding="utf-8-sig"
        )
    )


    compact_source = (
        "".join(
            source.split()
        )
    )


    assert (
        "ProxyHandler({})"
        in
        compact_source
    )



def test_redirect_is_blocked_before_destination(
) -> None:

    destination_hits = {
        "count":
            0
    }


    class DestinationHandler(
        QuietHandler
    ):
        def do_GET(
            self,
        ) -> None:

            destination_hits[
                "count"
            ] += 1

            self.send_response(
                200
            )

            self.end_headers()


        def do_POST(
            self,
        ) -> None:

            self.do_GET()


    destination_server = (
        HTTPServer(
            (
                "127.0.0.1",
                0,
            ),
            DestinationHandler,
        )
    )


    class RedirectHandler(
        QuietHandler
    ):
        def do_POST(
            self,
        ) -> None:

            target = (
                "http://127.0.0.1:"
                f"{destination_server.server_port}"
                "/capture"
            )

            self.send_response(
                302
            )

            self.send_header(
                "Location",
                target,
            )

            self.end_headers()


    redirect_server = (
        HTTPServer(
            (
                "127.0.0.1",
                0,
            ),
            RedirectHandler,
        )
    )


    destination_thread = Thread(
        target=
            destination_server
            .serve_forever,
        daemon=True,
    )

    redirect_thread = Thread(
        target=
            redirect_server
            .serve_forever,
        daemon=True,
    )


    destination_thread.start()
    redirect_thread.start()


    try:

        request = Request(
            (
                "http://127.0.0.1:"
                f"{redirect_server.server_port}"
                "/redirect"
            ),
            data=b"{}",
            headers={
                "Content-Type":
                    "application/json",
            },
            method="POST",
        )


        captured = None


        try:
            (
                llm_egress_module
                .open_local_llm_request(
                    request,
                    payload_class=(
                        LLMPayloadClass
                        .METADATA_ONLY
                    ),
                    timeout=2.0,
                )
            )

        except (
            llm_egress_module
            .LocalLLMEgressError
        ) as error:

            captured = (
                error
            )


        assert (
            captured
            is not None
        )


        assert (
            "redirect"
            in
            str(
                captured
            ).lower()
        )


        assert (
            destination_hits[
                "count"
            ]
            ==
            0
        )


    finally:

        redirect_server.shutdown()

        destination_server.shutdown()

        redirect_server.server_close()

        destination_server.server_close()


        redirect_thread.join(
            timeout=2.0
        )

        destination_thread.join(
            timeout=2.0
        )


def test_ollama_sdk_transport_is_hardened(
) -> None:

    inner = (
        provider_module
        .client
        ._client
    )


    assert (
        inner.follow_redirects
        is False
    )


    # Lock the DataLens provider wiring explicitly rather
    # than depending only on the installed Ollama defaults.
    provider_source = (
        Path(
            provider_module.__file__
        )
        .read_text(
            encoding="utf-8-sig"
        )
    )


    assert (
        "follow_redirects=False"
        in
        provider_source
    )


    assert (
        "trust_env=False"
        in
        provider_source
    )


def main(
) -> None:

    print(
        "=== DATALENS LLM REDIRECT "
        "EGRESS v0.1 ==="
    )

    print()


    tests = [
        (
            "Redirect egress rule version",
            test_rule_version,
        ),
        (
            "urllib environment proxies disabled",
            test_urllib_proxy_environment_is_disabled,
        ),
        (
            "Redirect blocked before destination",
            test_redirect_is_blocked_before_destination,
        ),
        (
            "Ollama SDK transport hardened",
            test_ollama_sdk_transport_is_hardened,
        ),
    ]


    for (
        label,
        test,
    ) in tests:

        test()

        print(
            f"[PASS] {label}"
        )


    print()

    print(
        "PASS - 4/4 LLM redirect "
        "egress security checks"
    )

    print(
        f"Rule: {TEST_RULE_VERSION}"
    )


if __name__ == "__main__":
    main()
