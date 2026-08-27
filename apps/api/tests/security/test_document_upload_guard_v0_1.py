from __future__ import annotations


from io import (
    BytesIO,
)

from unittest.mock import (
    patch,
)


from fastapi import (
    HTTPException,
    UploadFile,
)

from fastapi.testclient import (
    TestClient,
)


from main import (
    app,
)


import app.api.document_ingestion as document_ingestion_module

import app.rag as rag_module


# ============================================================
# VERSION
# ============================================================


TEST_RULE_VERSION = (
    "document_upload_guard_test_v0.1"
)


# ============================================================
# CLIENT
# ============================================================


client = TestClient(
    app
)


# ============================================================
# RECORDING STREAM
# ============================================================


class RecordingBytesIO(
    BytesIO
):
    """
    BytesIO variant that records every requested read size.

    The security contract requires the HTTP upload boundary
    to perform a bounded MAX_DOCUMENT_BYTES + 1 read rather
    than an unbounded read().
    """

    def __init__(
        self,
        content: bytes,
    ) -> None:
        super().__init__(
            content
        )

        self.read_sizes: list[
            int
        ] = []


    def read(
        self,
        size: int = -1,
    ) -> bytes:
        self.read_sizes.append(
            size
        )

        return super().read(
            size
        )


# ============================================================
# ASSERTION HELPERS
# ============================================================


def assert_equal(
    actual,
    expected,
    message: str,
) -> None:
    if (
        actual
        !=
        expected
    ):
        raise AssertionError(
            (
                f"{message}\n"
                f"expected={expected!r}\n"
                f"actual={actual!r}"
            )
        )


def assert_true(
    value,
    message: str,
) -> None:
    if not value:
        raise AssertionError(
            message
        )


# ============================================================
# UPLOAD HELPER
# ============================================================


def make_upload(
    *,
    filename: str,
    content: bytes,
) -> tuple[
    UploadFile,
    RecordingBytesIO,
]:
    stream = RecordingBytesIO(
        content
    )

    upload = UploadFile(
        filename=filename,
        file=stream,
    )

    return (
        upload,
        stream,
    )


# ============================================================
# TEST 1
# VERSION
# ============================================================


def test_document_upload_guard_version(
) -> None:
    assert_equal(
        (
            document_ingestion_module
            .DOCUMENT_UPLOAD_GUARD_RULE_VERSION
        ),
        "document_upload_guard_v0.1",
        (
            "Unexpected document upload "
            "guard version."
        ),
    )


# ============================================================
# TEST 2
# EXACT LIMIT
# ============================================================


def test_exact_limit_is_accepted_with_bounded_probe(
) -> None:
    (
        upload,
        stream,
    ) = make_upload(
        filename="limit.txt",
        content=b"12345678",
    )


    with patch.object(
        document_ingestion_module,
        "MAX_DOCUMENT_BYTES",
        8,
    ):
        documents = (
            document_ingestion_module
            .read_uploaded_documents(
                [
                    upload
                ]
            )
        )


    assert_equal(
        documents,
        [
            (
                "limit.txt",
                b"12345678",
            )
        ],
        (
            "A document exactly at the "
            "configured limit must be accepted."
        ),
    )


    assert_equal(
        stream.read_sizes,
        [
            9
        ],
        (
            "The upload boundary must request "
            "only MAX_DOCUMENT_BYTES + 1 bytes."
        ),
    )


    assert_true(
        stream.closed,
        (
            "Uploaded file stream must be "
            "closed after reading."
        ),
    )


# ============================================================
# TEST 3
# OVERSIZED BOUNDARY
# ============================================================


def test_oversized_upload_is_rejected_before_ingestion(
) -> None:
    ingestion_called = False


    def forbidden_ingestion(
        *,
        documents,
    ):
        nonlocal ingestion_called

        ingestion_called = True

        raise AssertionError(
            (
                "Oversized upload reached "
                "document ingestion."
            )
        )


    (
        upload,
        stream,
    ) = make_upload(
        filename="oversized.txt",
        content=b"123456789EXTRA",
    )


    captured_error = None


    with (
        patch.object(
            document_ingestion_module,
            "MAX_DOCUMENT_BYTES",
            8,
        ),
        patch.object(
            document_ingestion_module,
            "build_document_ingestion_report",
            forbidden_ingestion,
        ),
    ):
        try:
            (
                document_ingestion_module
                .ingest_document_uploads(
                    [
                        upload
                    ]
                )
            )

        except HTTPException as error:
            captured_error = error


    assert_true(
        captured_error is not None,
        (
            "Oversized upload must raise "
            "HTTPException."
        ),
    )


    assert_equal(
        captured_error.status_code,
        413,
        (
            "Oversized upload must return "
            "HTTP 413."
        ),
    )


    assert_equal(
        captured_error.detail,
        (
            document_ingestion_module
            .DOCUMENT_UPLOAD_TOO_LARGE_DETAIL
        ),
        (
            "Unexpected oversized-upload "
            "error detail."
        ),
    )


    assert_equal(
        stream.read_sizes,
        [
            9
        ],
        (
            "Oversized upload must still use "
            "a bounded limit + 1 read."
        ),
    )


    assert_true(
        stream.closed,
        (
            "Oversized upload stream must "
            "be closed."
        ),
    )


    assert_true(
        not ingestion_called,
        (
            "Oversized upload must be rejected "
            "before document ingestion."
        ),
    )


# ============================================================
# TEST 4
# HTTP CONTRACT
# ============================================================


def test_document_inspect_returns_413_for_oversized_upload(
) -> None:
    with patch.object(
        document_ingestion_module,
        "MAX_DOCUMENT_BYTES",
        8,
    ):
        response = client.post(
            "/rag/documents/inspect",
            files=[
                (
                    "document_files",
                    (
                        "oversized.txt",
                        b"123456789",
                        "text/plain",
                    ),
                )
            ],
        )


    assert_equal(
        response.status_code,
        413,
        (
            "/rag/documents/inspect must expose "
            "HTTP 413 for oversized uploads."
        ),
    )


    assert_equal(
        response.json(),
        {
            "detail":
                (
                    document_ingestion_module
                    .DOCUMENT_UPLOAD_TOO_LARGE_DETAIL
                )
        },
        (
            "Unexpected HTTP oversized-upload "
            "response."
        ),
    )


# ============================================================
# TEST 5
# DEFENSE IN DEPTH
# ============================================================


def test_core_document_processor_keeps_size_guard(
) -> None:
    captured_error = None


    with patch.object(
        rag_module,
        "MAX_DOCUMENT_BYTES",
        8,
    ):
        try:
            rag_module.process_document(
                filename="oversized.txt",
                content=b"123456789",
            )

        except ValueError as error:
            captured_error = error


    assert_true(
        captured_error is not None,
        (
            "Core document processor must "
            "independently reject oversized "
            "content."
        ),
    )


    assert_true(
        (
            "taille maximale"
            in
            str(
                captured_error
            )
        ),
        (
            "Core document size guard returned "
            "an unexpected error."
        ),
    )


# ============================================================
# RUNNER
# ============================================================


TESTS = [
    (
        "Document upload guard version",
        test_document_upload_guard_version,
    ),
    (
        "Exact size limit uses bounded read",
        test_exact_limit_is_accepted_with_bounded_probe,
    ),
    (
        "Oversized upload blocked before ingestion",
        test_oversized_upload_is_rejected_before_ingestion,
    ),
    (
        "HTTP endpoint returns 413",
        test_document_inspect_returns_413_for_oversized_upload,
    ),
    (
        "Core processor keeps defense in depth",
        test_core_document_processor_keeps_size_guard,
    ),
]


def main(
) -> None:
    print(
        "=== DATALENS DOCUMENT UPLOAD GUARD v0.1 ==="
    )

    print()


    passed = 0


    for (
        label,
        test,
    ) in TESTS:
        try:
            test()

        except Exception as error:
            print(
                f"[FAIL] {label}"
            )

            print(
                f"       {type(error).__name__}: "
                f"{error}"
            )

            raise


        passed += 1

        print(
            f"[PASS] {label}"
        )


    print()

    print(
        (
            f"PASS - {passed}/{len(TESTS)} "
            "document upload security checks"
        )
    )

    print(
        f"Rule: {TEST_RULE_VERSION}"
    )


if __name__ == "__main__":
    main()
