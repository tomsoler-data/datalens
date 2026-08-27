from __future__ import annotations


from fastapi.middleware.cors import (
    CORSMiddleware,
)

from fastapi.testclient import (
    TestClient,
)


from app.main import (
    LOCAL_FRONTEND_ALLOW_CREDENTIALS,
    LOCAL_FRONTEND_CORS_RULE_VERSION,
    LOCAL_FRONTEND_HEADERS,
    LOCAL_FRONTEND_METHODS,
    LOCAL_FRONTEND_ORIGINS,
    app,
)


TEST_RULE_VERSION = (
    "local_frontend_cors_test_v0.1"
)


def _cors_middleware(
):

    matching = [
        middleware

        for middleware
        in app.user_middleware

        if (
            middleware.cls
            is
            CORSMiddleware
        )
    ]


    assert (
        len(
            matching
        )
        ==
        1
    ), (
        "DataLens must own exactly one "
        "CORSMiddleware configuration."
    )


    return matching[
        0
    ]


def _preflight(
    *,
    origin: str,
    method: str,
    headers: str | None = None,
):

    request_headers = {
        "Origin":
            origin,

        "Access-Control-Request-Method":
            method,
    }


    if headers is not None:
        request_headers[
            "Access-Control-Request-Headers"
        ] = headers


    client = TestClient(
        app
    )


    return client.options(
        "/__cors_boundary_probe__",
        headers=
            request_headers,
    )


def test_rule_version(
) -> None:

    assert (
        LOCAL_FRONTEND_CORS_RULE_VERSION
        ==
        "local_frontend_cors_v0.1"
    )


def test_origins_are_exactly_local_frontends(
) -> None:

    assert (
        LOCAL_FRONTEND_ORIGINS
        ==
        [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )


    assert all(
        origin.startswith(
            (
                "http://localhost:",
                "http://127.0.0.1:",
            )
        )

        for origin
        in LOCAL_FRONTEND_ORIGINS
    )


def test_methods_are_minimal(
) -> None:

    assert (
        LOCAL_FRONTEND_METHODS
        ==
        [
            "GET",
            "POST",
            "DELETE",
        ]
    )


    assert (
        "*"
        not in
        LOCAL_FRONTEND_METHODS
    )


    assert (
        "PUT"
        not in
        LOCAL_FRONTEND_METHODS
    )


    assert (
        "PATCH"
        not in
        LOCAL_FRONTEND_METHODS
    )


def test_headers_are_minimal(
) -> None:

    assert (
        LOCAL_FRONTEND_HEADERS
        ==
        [
            "Content-Type",
        ]
    )


    assert (
        "*"
        not in
        LOCAL_FRONTEND_HEADERS
    )


    assert (
        "Authorization"
        not in
        LOCAL_FRONTEND_HEADERS
    )


def test_credentials_are_disabled_and_wiring_is_locked(
) -> None:

    assert (
        LOCAL_FRONTEND_ALLOW_CREDENTIALS
        is
        False
    )


    middleware = (
        _cors_middleware()
    )


    assert (
        middleware.kwargs[
            "allow_origins"
        ]
        ==
        LOCAL_FRONTEND_ORIGINS
    )


    assert (
        middleware.kwargs[
            "allow_methods"
        ]
        ==
        LOCAL_FRONTEND_METHODS
    )


    assert (
        middleware.kwargs[
            "allow_headers"
        ]
        ==
        LOCAL_FRONTEND_HEADERS
    )


    assert (
        middleware.kwargs[
            "allow_credentials"
        ]
        is
        False
    )


def test_allowed_local_preflight_succeeds(
) -> None:

    response = (
        _preflight(
            origin=
                "http://localhost:3000",

            method=
                "POST",

            headers=
                "content-type",
        )
    )


    assert (
        response.status_code
        ==
        200
    )


    assert (
        response.headers.get(
            "access-control-allow-origin"
        )
        ==
        "http://localhost:3000"
    )


    allow_headers = (
        response.headers.get(
            "access-control-allow-headers",
            "",
        )
        .lower()
    )


    assert (
        "content-type"
        in
        allow_headers
    )


    assert (
        response.headers.get(
            "access-control-allow-credentials"
        )
        is None
    )


def test_remote_method_and_header_expansion_fail_closed(
) -> None:

    remote = (
        _preflight(
            origin=
                "https://example.com",

            method=
                "POST",

            headers=
                "content-type",
        )
    )


    assert (
        remote.status_code
        ==
        400
    )


    assert (
        remote.headers.get(
            "access-control-allow-origin"
        )
        is None
    )


    patch_request = (
        _preflight(
            origin=
                "http://127.0.0.1:3000",

            method=
                "PATCH",

            headers=
                "content-type",
        )
    )


    assert (
        patch_request.status_code
        ==
        400
    )


    authorization = (
        _preflight(
            origin=
                "http://127.0.0.1:3000",

            method=
                "POST",

            headers=
                "authorization",
        )
    )


    assert (
        authorization.status_code
        ==
        400
    )


def main(
) -> None:

    print(
        "=== DATALENS LOCAL FRONTEND "
        "CORS v0.1 ==="
    )

    print()


    tests = [
        (
            "CORS rule version",
            test_rule_version,
        ),
        (
            "Origins exactly local",
            test_origins_are_exactly_local_frontends,
        ),
        (
            "Methods minimal",
            test_methods_are_minimal,
        ),
        (
            "Headers minimal",
            test_headers_are_minimal,
        ),
        (
            "Credentials disabled and wiring locked",
            test_credentials_are_disabled_and_wiring_is_locked,
        ),
        (
            "Allowed local preflight succeeds",
            test_allowed_local_preflight_succeeds,
        ),
        (
            "Expansion attempts fail closed",
            test_remote_method_and_header_expansion_fail_closed,
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
        "PASS - 7/7 local frontend "
        "CORS checks"
    )

    print(
        f"Rule: {TEST_RULE_VERSION}"
    )


if __name__ == "__main__":
    main()
