from __future__ import annotations


from collections import (
    Counter,
)


from app.document_summary import (
    build_deterministic_claims,
    build_document_evidence_catalog,
)

from app.planning.request_planner import (
    classify_request,
)

from app.rag import (
    build_document_ingestion_report,
)


# ============================================================
# SOURCE DOCUMENTS
# ============================================================

JULIE_DOCUMENT = """
Conversation avec Julie

Moi
Salut Julie, je suis le nouveau Data Analyst chargé d'analyser
les données de nos ventes en ligne. Annabelle m'a dit que tu avais
des demandes spécifiques concernant l'analyse des comportements
clients.

Julie
Salut !
Bienvenue dans l'équipe ! J'allais justement prendre contact avec
toi car je vais avoir besoin que tu regardes certaines corrélations.
Nous souhaitons mieux comprendre le comportement de nos clients.

Moi
Bien sûr. Tu as déjà une idée précise de ce que tu veux que je
regarde ?

Julie
Oui ! J'aimerais que tu regardes en particulier 5 corrélations :
● le lien entre le genre d'un client et les catégories des livres achetés,
● le lien entre l'âge des clients et le montant total des achats,
● le lien entre l'âge des clients et la fréquence d'achat,
● le lien entre l'âge des clients et la taille du panier moyen,
● le lien entre l'âge des clients et la catégorie des livres achetés.

Moi
Ça marche, je m'occupe de ça et je reviens vers toi si j'ai des
questions ou pour te présenter les résultats !
""".strip()


BRIEF_DOCUMENT = """
Brief de l'analyse

Nous souhaitons élaborer différents graphiques autour du chiffre
d'affaires comme par exemples l'évolution dans le temps du :
● chiffre d'affaires avec la moyenne mobile
  (tu pourras choisir la période : jour, semaine, mois, etc.),
● chiffre d'affaires par catégorie,
● nombre de clients par mois,
● nombre de transactions,
● nombre de produits vendus,
● etc.

Il serait également intéressant de faire un zoom sur les références :
● les tops,
● les flops,
● la répartition par catégorie,
● etc.

Enfin, j'aimerais avoir quelques informations sur les profils de nos clients :
● répartition du chiffre d'affaires pour les clients BtoB,
● courbe de Lorenz,
● etc.

Après, toutes les informations et tous graphiques qui apporteraient
de l'information pertinente sont les bienvenus !
""".strip()


# ============================================================
# EXPECTATIONS
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


EXPECTED_ALL_KINDS = (
    EXPECTED_JULIE_KINDS
    |
    EXPECTED_BRIEF_KINDS
)


# ============================================================
# ASSERTIONS
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
# EXTRACTION
# ============================================================

def extract_requests():
    ingestion = (
        build_document_ingestion_report(
            documents=[
                (
                    "conversation avec Julie.txt",
                    JULIE_DOCUMENT.encode(
                        "utf-8"
                    ),
                ),
                (
                    "Brief de l'analyse.txt",
                    BRIEF_DOCUMENT.encode(
                        "utf-8"
                    ),
                ),
            ]
        )
    )


    catalog = (
        build_document_evidence_catalog(
            ingestion
        )
    )


    requests = []


    for document in ingestion.documents:
        units = (
            catalog.get(
                document.document_id,
                [],
            )
        )


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


            requests.append(
                (
                    claim,
                    classify_request(
                        claim
                    ),
                )
            )


    return (
        ingestion,
        requests,
    )


# ============================================================
# TESTS
# ============================================================

def test_document_ingestion() -> None:
    (
        ingestion,
        _,
    ) = extract_requests()


    assert_equal(
        ingestion.document_count,
        2,
        "Two Lapage documents should be ingested.",
    )


    pass_test(
        "two Lapage source documents are ingested"
    )


def test_exact_request_count() -> None:
    (
        _,
        requests,
    ) = extract_requests()


    assert_equal(
        len(
            requests
        ),
        15,
        (
            "The Python-first inventory should "
            "detect exactly 15 concrete requests."
        ),
    )


    pass_test(
        "15 concrete analytical requests are detected"
    )


def test_per_document_counts() -> None:
    (
        _,
        requests,
    ) = extract_requests()


    counts = (
        Counter(
            claim
            .citation
            .filename

            for (
                claim,
                _,
            )
            in requests
        )
    )


    assert_equal(
        counts[
            "conversation avec Julie.txt"
        ],
        5,
        (
            "Julie document should expose exactly "
            "5 analytical requests."
        ),
    )


    assert_equal(
        counts[
            "Brief de l'analyse.txt"
        ],
        10,
        (
            "Brief should expose exactly "
            "10 concrete analytical requests."
        ),
    )


    pass_test(
        "Julie contributes 5 requests and brief contributes 10"
    )


def test_all_expected_kinds() -> None:
    (
        _,
        requests,
    ) = extract_requests()


    detected_kinds = {
        kind

        for (
            _,
            kind,
        )
        in requests
    }


    missing = (
        EXPECTED_ALL_KINDS
        -
        detected_kinds
    )


    unexpected = (
        detected_kinds
        -
        EXPECTED_ALL_KINDS
    )


    assert_equal(
        missing,
        set(),
        (
            "Some expected Lapage analytical "
            "families were not classified."
        ),
    )


    assert_equal(
        unexpected,
        set(),
        (
            "Unexpected analytical families were "
            "created by the inventory."
        ),
    )


    assert_equal(
        len(
            detected_kinds
        ),
        15,
        (
            "Each of the 15 benchmark requests "
            "should map to a distinct expected kind."
        ),
    )


    pass_test(
        "all 15 expected analytical kinds are classified"
    )


def test_no_unknown_request() -> None:
    (
        _,
        requests,
    ) = extract_requests()


    unknown = [
        claim.statement

        for (
            claim,
            kind,
        )
        in requests

        if (
            kind
            ==
            "unknown"
        )
    ]


    assert_equal(
        unknown,
        [],
        (
            "No benchmark request should remain "
            "unknown."
        ),
    )


    pass_test(
        "no Lapage request remains unknown"
    )


def test_etc_is_not_a_request() -> None:
    (
        _,
        requests,
    ) = extract_requests()


    invalid = [
        claim.statement

        for (
            claim,
            _,
        )
        in requests

        if (
            claim.statement
            .strip()
            .casefold()
            in {
                "etc",
                "etc.",
            }
        )
    ]


    assert_equal(
        invalid,
        [],
        (
            "'etc.' must terminate a list and must "
            "not become an analytical request."
        ),
    )


    pass_test(
        "'etc.' is not converted into an analytical request"
    )


def test_open_invitation_is_not_explicit_request() -> None:
    (
        _,
        requests,
    ) = extract_requests()


    invalid = [
        claim.statement

        for (
            claim,
            _,
        )
        in requests

        if (
            "toutes les informations"
            in
            claim.statement.casefold()
            or
            "tous graphiques"
            in
            claim.statement.casefold()
        )
    ]


    assert_equal(
        invalid,
        [],
        (
            "The broad invitation to add relevant "
            "information must not be represented as "
            "an explicit analytical request."
        ),
    )


    pass_test(
        (
            "general invitation for additional charts "
            "is not treated as explicit"
        )
    )


# ============================================================
# DIAGNOSTIC OUTPUT
# ============================================================

def print_inventory() -> None:
    (
        _,
        requests,
    ) = extract_requests()


    print()

    print(
        "===== DETECTED REQUEST INVENTORY ====="
    )


    for (
        index,
        (
            claim,
            kind,
        ),
    ) in enumerate(
        requests,
        start=1,
    ):
        print(
            (
                f"{index:02d}. "
                f"[{claim.citation.filename}] "
                f"{kind}"
            )
        )

        print(
            f"    {claim.statement}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "=== DATALENS LAPAGE REQUEST INVENTORY v0.1 ==="
    )

    print()


    test_document_ingestion()

    test_exact_request_count()

    test_per_document_counts()

    test_all_expected_kinds()

    test_no_unknown_request()

    test_etc_is_not_a_request()

    test_open_invitation_is_not_explicit_request()


    print_inventory()


    print()

    print(
        "PASS - Lapage request inventory v0.1"
    )


if (
    __name__
    ==
    "__main__"
):
    main()