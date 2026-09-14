from __future__ import annotations


from types import (
    SimpleNamespace,
)


from app.rag_context import (
    RAG_CONTEXT_RULE_VERSION,
    build_analytical_contract,
    build_analytical_contract_text,
)

from app.rag_relevance import (
    RELEVANCE_RULE_VERSION,
    build_evidence_units,
    filter_candidate_evidence_units,
)


def categorical_finding_text() -> str:
    return """
CONTRAT ANALYTIQUE D\u00c9TERMINISTE
Famille analytique : categorical_association
Variable X : gender
Variable Y : category
Relation analytique requise : \u00c9tudier la relation analytique entre `gender` et `category`.
""".strip()


def test_categorical_association_requires_x_and_y() -> None:
    finding = (
        categorical_finding_text()
    )

    relevant_units = (
        build_evidence_units(
            (
                "The gender of customers can be studied "
                "according to the category of products "
                "purchased."
            )
        )
    )

    irrelevant_units = (
        build_evidence_units(
            (
                "Une corr\u00e9lation ne d\u00e9montre pas "
                "\u00e0 elle seule une relation causale. "
                "La m\u00e9diane est moins sensible aux "
                "valeurs extr\u00eames que la moyenne."
            )
        )
    )

    relevant_candidates = (
        filter_candidate_evidence_units(
            finding=
                finding,

            evidence_units=
                relevant_units,
        )
    )

    irrelevant_candidates = (
        filter_candidate_evidence_units(
            finding=
                finding,

            evidence_units=
                irrelevant_units,
        )
    )

    assert len(
        relevant_candidates
    ) == 1

    assert (
        irrelevant_candidates
        ==
        []
    )


def test_derived_gap_contract_is_structured() -> None:
    finding = SimpleNamespace(
        title=(
            "\u00c9cart entre Target canopy temperature (C) "
            "et Measured canopy temperature (C)"
        ),
        family=
            "derived_gap",

        metrics={
            "left_column":
                "Target canopy temperature (C)",

            "right_column":
                "Measured canopy temperature (C)",
        },
    )

    contract = (
        build_analytical_contract(
            finding=
                finding,
        )
    )

    assert (
        contract.left_column
        ==
        "Target canopy temperature (C)"
    )

    assert (
        contract.right_column
        ==
        "Measured canopy temperature (C)"
    )

    contract_text = (
        build_analytical_contract_text(
            contract=
                contract,
        )
    )

    assert (
        "Mesure gauche : "
        "Target canopy temperature (C)"
        in
        contract_text
    )

    assert (
        "Mesure droite : "
        "Measured canopy temperature (C)"
        in
        contract_text
    )

    relevant_candidates = (
        filter_candidate_evidence_units(
            finding=
                contract_text,

            evidence_units=
                build_evidence_units(
                    (
                        "Target canopy temperature (C) "
                        "et Measured canopy temperature (C) "
                        "doivent \u00eatre compar\u00e9es pour "
                        "\u00e9tudier leur \u00e9cart."
                    )
                ),
        )
    )

    irrelevant_candidates = (
        filter_candidate_evidence_units(
            finding=
                contract_text,

            evidence_units=
                build_evidence_units(
                    (
                        "La m\u00e9diane est moins sensible "
                        "aux valeurs extr\u00eames que la "
                        "moyenne."
                    )
                ),
        )
    )

    assert len(
        relevant_candidates
    ) == 1

    assert (
        irrelevant_candidates
        ==
        []
    )


def test_unstructured_unknown_family_fails_closed() -> None:
    finding = """
CONTRAT ANALYTIQUE D\u00c9TERMINISTE
Famille analytique : documentary_context
Relation analytique requise : Fournir un contexte documentaire.
""".strip()

    candidates = (
        filter_candidate_evidence_units(
            finding=
                finding,

            evidence_units=
                build_evidence_units(
                    (
                        "Une corr\u00e9lation ne d\u00e9montre "
                        "pas \u00e0 elle seule une causalit\u00e9."
                    )
                ),
        )
    )

    assert (
        candidates
        ==
        []
    )


def test_rule_versions() -> None:
    assert (
        RELEVANCE_RULE_VERSION
        ==
        "rag_relevance_v0.9"
    )

    assert (
        RAG_CONTEXT_RULE_VERSION
        ==
        "rag_contextualization_v0.7"
    )


if __name__ == "__main__":
    test_categorical_association_requires_x_and_y()

    print(
        "[PASS] categorical association requires X and Y"
    )

    test_derived_gap_contract_is_structured()

    print(
        "[PASS] derived gap carries left and right measures"
    )

    test_unstructured_unknown_family_fails_closed()

    print(
        "[PASS] unstructured unknown family fails closed"
    )

    test_rule_versions()

    print(
        "[PASS] RAG relevance/context rule versions"
    )

    print()

    print(
        "PASS - RAG relevance contract guard v0.1"
    )
