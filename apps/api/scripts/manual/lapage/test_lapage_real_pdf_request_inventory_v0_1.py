from __future__ import annotations


from collections import (
    Counter,
)

from pathlib import (
    Path,
)


from app.rag import (
    build_document_ingestion_report,
)

from app.document_summary import (
    build_document_evidence_catalog,
    build_deterministic_claims,
)

from app.planning.request_planner import (
    classify_request,
)


# ============================================================
# REAL LAPAGE FILES
# ============================================================

BRIEF_PATH = Path(
    r"C:\Users\tomas\Downloads\Projet_9\Projet_9\Documents\Brief+de+l'analyse.pdf"
)

JULIE_PATH = Path(
    r"C:\Users\tomas\Downloads\Projet_9\Projet_9\Documents\conversation+avec+Julie+.pdf"
)


BRIEF_FILENAME = (
    "Brief de l'analyse.pdf"
)

JULIE_FILENAME = (
    "conversation avec Julie.pdf"
)


# ============================================================
# EXPECTED REQUEST INVENTORY
# ============================================================

EXPECTED_JULIE_KINDS = {
    "gender_category_association",
    "age_total_amount_association",
    "age_frequency_association",
    "age_average_basket_association",
    "age_category_association",
}


EXPECTED_BRIEF_KINDS = {
    "revenue_moving_average",
    "revenue_by_category",
    "customers_by_period",
    "transaction_count",
    "products_sold_count",
    "top_products",
    "flop_products",
    "product_category_distribution",
    "b2b_revenue_distribution",
    "lorenz_curve",
}


EXPECTED_KINDS = (
    EXPECTED_JULIE_KINDS
    |
    EXPECTED_BRIEF_KINDS
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
                f"Expected: {expected!r}\n"
                f"Actual:   {actual!r}"
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


def pass_test(
    message: str,
) -> None:
    print(
        f"[PASS] {message}"
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(
    value: object,
) -> str:
    return (
        " ".join(
            str(
                value
                or
                ""
            )
            .replace(
                "\u00a0",
                " ",
            )
            .replace(
                "\u202f",
                " ",
            )
            .split()
        )
        .strip()
        .lower()
    )


# ============================================================
# BUILD REAL PDF INVENTORY
# ============================================================

def build_real_pdf_inventory():
    assert_true(
        BRIEF_PATH.exists(),
        (
            "The real Lapage brief PDF "
            "could not be found."
        ),
    )


    assert_true(
        JULIE_PATH.exists(),
        (
            "The real Julie PDF "
            "could not be found."
        ),
    )


    ingestion = (
        build_document_ingestion_report(
            documents=[
                (
                    BRIEF_FILENAME,
                    BRIEF_PATH.read_bytes(),
                ),
                (
                    JULIE_FILENAME,
                    JULIE_PATH.read_bytes(),
                ),
            ]
        )
    )


    catalog = (
        build_document_evidence_catalog(
            ingestion
        )
    )


    analytical_claims = []


    for units in (
        catalog.values()
    ):
        claims = (
            build_deterministic_claims(
                units
            )
        )


        for claim in claims:
            if (
                claim.category
                !=
                "analytical_request"
            ):
                continue


            analytical_claims.append(
                claim
            )


    inventory = []


    for claim in (
        analytical_claims
    ):
        kind = (
            classify_request(
                claim
            )
        )


        inventory.append(
            {
                "claim":
                    claim,

                "kind":
                    kind,

                "filename":
                    claim
                    .citation
                    .filename,

                "source_locator":
                    claim
                    .citation
                    .source_locator,

                "page_number":
                    claim
                    .citation
                    .page_number,

                "evidence_unit_id":
                    claim
                    .evidence_unit_id,

                "evidence_quote":
                    claim
                    .evidence_quote,

                "context_quote":
                    claim
                    .context_quote,
            }
        )


    return (
        ingestion,
        catalog,
        inventory,
    )


# ============================================================
# TEST 1 — REAL FILES INGEST
# ============================================================

def test_real_pdfs_are_ingested(
) -> None:
    (
        ingestion,
        _,
        _,
    ) = build_real_pdf_inventory()


    assert_equal(
        ingestion.document_count,
        2,
        (
            "Exactly two real Lapage PDF "
            "documents should be ingested."
        ),
    )


    pass_test(
        "two real Lapage PDFs are ingested"
    )


# ============================================================
# TEST 2 — EXACT REQUEST COUNT
# ============================================================

def test_real_pdfs_produce_15_requests(
) -> None:
    (
        _,
        _,
        inventory,
    ) = build_real_pdf_inventory()


    assert_equal(
        len(
            inventory
        ),
        15,
        (
            "The two real Lapage PDFs must "
            "produce exactly 15 concrete "
            "analytical requests."
        ),
    )


    pass_test(
        (
            "real Lapage PDFs produce exactly "
            "15 analytical requests"
        )
    )


# ============================================================
# TEST 3 — REQUEST COUNT BY DOCUMENT
# ============================================================

def test_real_pdf_document_distribution(
) -> None:
    (
        _,
        _,
        inventory,
    ) = build_real_pdf_inventory()


    counts = (
        Counter(
            item[
                "filename"
            ]

            for item
            in inventory
        )
    )


    assert_equal(
        counts[
            JULIE_FILENAME
        ],
        5,
        (
            "The real Julie PDF must contribute "
            "exactly five analytical requests."
        ),
    )


    assert_equal(
        counts[
            BRIEF_FILENAME
        ],
        10,
        (
            "The real brief PDF must contribute "
            "exactly ten analytical requests."
        ),
    )


    pass_test(
        (
            "real PDFs preserve "
            "5 Julie + 10 brief requests"
        )
    )


# ============================================================
# TEST 4 — EXACT 15 KINDS
# ============================================================

def test_real_pdf_exact_kinds(
) -> None:
    (
        _,
        _,
        inventory,
    ) = build_real_pdf_inventory()


    actual_kinds = {
        item[
            "kind"
        ]

        for item
        in inventory
    }


    assert_equal(
        actual_kinds,
        EXPECTED_KINDS,
        (
            "The real PDFs did not produce the "
            "exact 15 expected analytical kinds."
        ),
    )


    assert_equal(
        len(
            actual_kinds
        ),
        15,
        (
            "The real PDF inventory must contain "
            "15 distinct expected request kinds."
        ),
    )


    pass_test(
        (
            "real PDFs preserve all "
            "15 expected analytical kinds"
        )
    )


# ============================================================
# TEST 5 — EXACT DOCUMENT KIND DISTRIBUTION
# ============================================================

def test_real_pdf_kinds_by_document(
) -> None:
    (
        _,
        _,
        inventory,
    ) = build_real_pdf_inventory()


    julie_kinds = {
        item[
            "kind"
        ]

        for item
        in inventory

        if (
            item[
                "filename"
            ]
            ==
            JULIE_FILENAME
        )
    }


    brief_kinds = {
        item[
            "kind"
        ]

        for item
        in inventory

        if (
            item[
                "filename"
            ]
            ==
            BRIEF_FILENAME
        )
    }


    assert_equal(
        julie_kinds,
        EXPECTED_JULIE_KINDS,
        (
            "The real Julie PDF did not preserve "
            "its five expected request kinds."
        ),
    )


    assert_equal(
        brief_kinds,
        EXPECTED_BRIEF_KINDS,
        (
            "The real brief PDF did not preserve "
            "its ten expected request kinds."
        ),
    )


    pass_test(
        (
            "request kinds remain attached to "
            "the correct real source PDF"
        )
    )


# ============================================================
# TEST 6 — NO UNKNOWN
# ============================================================

def test_real_pdf_has_no_unknown_request(
) -> None:
    (
        _,
        _,
        inventory,
    ) = build_real_pdf_inventory()


    unknown = [
        item

        for item
        in inventory

        if (
            item[
                "kind"
            ]
            ==
            "unknown"
        )
    ]


    assert_equal(
        unknown,
        [],
        (
            "One or more real PDF requests "
            "remain classified as unknown."
        ),
    )


    pass_test(
        (
            "no real Lapage PDF request "
            "remains unknown"
        )
    )


# ============================================================
# TEST 7 — ETC IS NOT A STANDALONE REQUEST
# ============================================================

def test_real_pdf_etc_is_not_standalone_request(
) -> None:
    (
        _,
        _,
        inventory,
    ) = build_real_pdf_inventory()


    false_etc_requests = []


    for item in (
        inventory
    ):
        quote = (
            normalize_text(
                item[
                    "evidence_quote"
                ]
            )
        )


        if quote in {
            "etc",
            "etc.",
            "etc…",
            "etc...",
        }:
            false_etc_requests.append(
                item
            )


    assert_equal(
        false_etc_requests,
        [],
        (
            "A standalone 'etc.' extracted from "
            "the real brief PDF was incorrectly "
            "treated as an analytical request."
        ),
    )


    pass_test(
        (
            "standalone 'etc.' is not converted "
            "into a real PDF analytical request"
        )
    )


# ============================================================
# TEST 8 — GENERIC INVITATION IS NOT EXPLICIT REQUEST
# ============================================================

def test_generic_invitation_is_not_request(
) -> None:
    (
        _,
        _,
        inventory,
    ) = build_real_pdf_inventory()


    false_generic_requests = []


    markers = (
        "toutes les informations",
        "tous graphiques qui apporteraient",
        "graphiques qui apporteraient",
        "information pertinente sont les bienvenus",
    )


    for item in (
        inventory
    ):
        quote = (
            normalize_text(
                item[
                    "evidence_quote"
                ]
            )
        )


        if any(
            marker
            in
            quote

            for marker
            in markers
        ):
            false_generic_requests.append(
                item
            )


    assert_equal(
        false_generic_requests,
        [],
        (
            "The generic invitation for additional "
            "information or charts was incorrectly "
            "classified as an explicit analytical "
            "request in the real PDF."
        ),
    )


    pass_test(
        (
            "generic invitation for additional "
            "charts is not treated as explicit"
        )
    )


# ============================================================
# TEST 9 — DOCUMENTARY PROVENANCE EXISTS
# ============================================================

def test_real_pdf_provenance_is_preserved(
) -> None:
    (
        _,
        _,
        inventory,
    ) = build_real_pdf_inventory()


    for item in (
        inventory
    ):
        assert_true(
            bool(
                item[
                    "filename"
                ]
            ),
            (
                "A real PDF request lost its "
                "source filename."
            ),
        )


        assert_true(
            bool(
                item[
                    "source_locator"
                ]
            ),
            (
                "A real PDF request lost its "
                "source locator."
            ),
        )


        assert_true(
            bool(
                item[
                    "evidence_quote"
                ]
            ),
            (
                "A real PDF request lost its "
                "evidence quote."
            ),
        )


        assert_true(
            item[
                "evidence_unit_id"
            ]
            is not None,
            (
                "A real PDF request lost its "
                "evidence-unit identity."
            ),
        )


    pass_test(
        (
            "all 15 real PDF requests preserve "
            "documentary provenance"
        )
    )


# ============================================================
# DIAGNOSTIC OUTPUT
# ============================================================

def print_real_pdf_inventory(
) -> None:
    (
        ingestion,
        catalog,
        inventory,
    ) = build_real_pdf_inventory()


    print()

    print(
        "===== REAL PDF INGESTION ====="
    )

    print()


    print(
        (
            "documents: "
            f"{ingestion.document_count}"
        )
    )


    print(
        (
            "chunks: "
            f"{ingestion.chunk_count}"
        )
    )


    print(
        (
            "evidence catalogs: "
            f"{len(catalog)}"
        )
    )


    print()

    print(
        "===== REAL PDF REQUEST INVENTORY ====="
    )

    print()


    for (
        index,
        item,
    ) in enumerate(
        inventory,
        start=1,
    ):
        print(
            (
                f"{index:02d}. "
                f"[{item['kind']}]"
            )
        )


        print(
            (
                f"    source: "
                f"{item['filename']}"
            )
        )


        print(
            (
                f"    locator: "
                f"{item['source_locator']}"
            )
        )


        print(
            (
                f"    page: "
                f"{item['page_number']}"
            )
        )


        print(
            (
                f"    evidence unit: "
                f"{item['evidence_unit_id']}"
            )
        )


        print(
            (
                f"    evidence: "
                f"{item['evidence_quote']}"
            )
        )


        if (
            item[
                "context_quote"
            ]
        ):
            print(
                (
                    f"    context: "
                    f"{item['context_quote']}"
                )
            )


        print()


    counts = (
        Counter(
            item[
                "filename"
            ]

            for item
            in inventory
        )
    )


    print(
        "===== REAL PDF SUMMARY ====="
    )

    print()


    print(
        (
            f"{JULIE_FILENAME}: "
            f"{counts[JULIE_FILENAME]}"
        )
    )


    print(
        (
            f"{BRIEF_FILENAME}: "
            f"{counts[BRIEF_FILENAME]}"
        )
    )


    print(
        (
            "total: "
            f"{len(inventory)}"
        )
    )


    print(
        (
            "unknown: "
            f"{sum(
                1
                for item
                in inventory
                if item['kind'] == 'unknown'
            )}"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        (
            "=== DATALENS LAPAGE REAL PDF "
            "REQUEST INVENTORY v0.1 ==="
        )
    )

    print()


    test_real_pdfs_are_ingested()

    test_real_pdfs_produce_15_requests()

    test_real_pdf_document_distribution()

    test_real_pdf_exact_kinds()

    test_real_pdf_kinds_by_document()

    test_real_pdf_has_no_unknown_request()

    test_real_pdf_etc_is_not_standalone_request()

    test_generic_invitation_is_not_request()

    test_real_pdf_provenance_is_preserved()


    print_real_pdf_inventory()


    print()

    print(
        (
            "PASS - Lapage real PDF "
            "request inventory v0.1"
        )
    )


if (
    __name__
    ==
    "__main__"
):
    main()