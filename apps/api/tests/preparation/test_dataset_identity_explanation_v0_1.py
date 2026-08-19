from __future__ import annotations

import json

import pandas as pd

from app.preparation.dataset_identity import (
    profile_dataset_identity,
)

from app.preparation.dataset_identity_explanation import (
    DATASET_IDENTITY_EXPLANATION_RULE_VERSION,
    RawDatasetIdentityExplanation,
    build_identity_explanation_facts,
    build_identity_explanation_prompt,
    validate_identity_explanation,
)


def reliable_key_report():
    dataframe = pd.DataFrame(
        {
            "order_id": [
                "O1",
                "O2",
                "O3",
            ],
            "customer_id": [
                "C1",
                "C1",
                "C2",
            ],
            "amount": [
                10.0,
                20.0,
                30.0,
            ],
        }
    )

    return profile_dataset_identity(
        dataframe,
        dataset_id="orders",
        dataset_filename="orders.csv",
    )


def surrogate_report():
    dataframe = pd.DataFrame(
        {
            "city": [
                "Paris",
                "Paris",
                "Lyon",
            ],
            "value": [
                10,
                10,
                20,
            ],
        }
    )

    return profile_dataset_identity(
        dataframe,
        dataset_id="observations",
        dataset_filename="observations.csv",
    )


def test_prompt_contains_only_structured_facts() -> None:
    report = surrogate_report()

    facts = build_identity_explanation_facts(
        report
    )

    assert "row_count" in facts
    assert "preferred_candidate" in facts
    assert "dataframe" not in facts
    assert "rows" not in facts

    prompt = build_identity_explanation_prompt(
        report
    )

    parsed_facts = json.dumps(
        facts,
        ensure_ascii=False,
    )

    assert "surrogate_recommended" in prompt
    assert "row_id" in parsed_facts

    print(
        "Identity explanation uses structured facts only: PASS"
    )


def test_valid_surrogate_explanation() -> None:
    report = surrogate_report()

    raw = RawDatasetIdentityExplanation(
        action="create_surrogate_key",
        confidence=0.94,
        title="Identifiant technique recommandé",
        explanation=(
            "Aucune clé naturelle fiable n'a été "
            "détectée par Python."
        ),
        user_message=(
            "Créer row_id améliorera la traçabilité "
            "des observations."
        ),
        referenced_columns=[],
        surrogate_column="row_id",
        cautions=[
            (
                "Cette clé technique ne doit pas être "
                "utilisée comme clé de jointure."
            )
        ],
    )

    validated = validate_identity_explanation(
        report=report,
        raw=raw,
    )

    assert validated.action == "create_surrogate_key"
    assert validated.surrogate_column == "row_id"
    assert validated.python_validated is True
    assert validated.executable is False
    assert validated.requires_user_confirmation is True

    print(
        "Valid surrogate explanation accepted: PASS"
    )


def test_model_cannot_invent_surrogate_name() -> None:
    report = surrogate_report()

    raw = RawDatasetIdentityExplanation(
        action="create_surrogate_key",
        confidence=0.9,
        title="Créer une clé",
        explanation="Clé technique suggérée.",
        user_message="Créer technical_pk.",
        referenced_columns=[],
        surrogate_column="technical_pk",
        cautions=[],
    )

    try:
        validate_identity_explanation(
            report=report,
            raw=raw,
        )

    except ValueError as error:
        assert "invent or change" in str(
            error
        )

    else:
        raise AssertionError(
            (
                "Invented surrogate column "
                "should have been rejected."
            )
        )

    print(
        "Invented surrogate column rejected: PASS"
    )


def test_model_cannot_replace_reliable_key_with_surrogate(
) -> None:
    report = reliable_key_report()

    raw = RawDatasetIdentityExplanation(
        action="create_surrogate_key",
        confidence=0.7,
        title="Créer row_id",
        explanation=(
            "Le modèle préfère une clé technique."
        ),
        user_message="Créer une nouvelle clé.",
        referenced_columns=[
            "order_id",
        ],
        surrogate_column="row_id",
        cautions=[],
    )

    try:
        validate_identity_explanation(
            report=report,
            raw=raw,
        )

    except ValueError as error:
        assert "already detected" in str(
            error
        )

    else:
        raise AssertionError(
            (
                "LLM override of reliable key "
                "should have been rejected."
            )
        )

    print(
        "Reliable Python key cannot be overridden by LLM: PASS"
    )


def test_model_cannot_reference_unknown_column() -> None:
    report = reliable_key_report()

    raw = RawDatasetIdentityExplanation(
        action="keep_detected_key",
        confidence=0.95,
        title="Conserver order_id",
        explanation="Python a validé order_id.",
        user_message="Conserver la clé détectée.",
        referenced_columns=[
            "invented_customer_key",
        ],
        surrogate_column=None,
        cautions=[],
    )

    try:
        validate_identity_explanation(
            report=report,
            raw=raw,
        )

    except ValueError as error:
        assert "unknown or unauthorized" in str(
            error
        )

    else:
        raise AssertionError(
            (
                "Unknown referenced column "
                "should have been rejected."
            )
        )

    print(
        "Unknown LLM column reference rejected: PASS"
    )


def test_valid_detected_key_explanation() -> None:
    report = reliable_key_report()

    raw = RawDatasetIdentityExplanation(
        action="keep_detected_key",
        confidence=0.98,
        title="Clé de ligne détectée",
        explanation=(
            "order_id est complet, unique et possède "
            "un signal explicite d'identifiant."
        ),
        user_message=(
            "Aucune clé technique supplémentaire "
            "n'est nécessaire."
        ),
        referenced_columns=[
            "order_id",
        ],
        surrogate_column=None,
        cautions=[
            (
                "Cette conclusion concerne l'identité des lignes, "
                "pas automatiquement les relations avec d'autres tables."
            )
        ],
    )

    validated = validate_identity_explanation(
        report=report,
        raw=raw,
    )

    assert validated.action == "keep_detected_key"
    assert validated.referenced_columns == [
        "order_id",
    ]
    assert validated.surrogate_column is None
    assert validated.requires_user_confirmation is False

    print(
        "Valid detected-key explanation accepted: PASS"
    )


def main() -> None:
    print(
        "=== DATALENS DATASET IDENTITY EXPLANATION v0.1 ==="
    )

    print()

    test_prompt_contains_only_structured_facts()
    test_valid_surrogate_explanation()
    test_model_cannot_invent_surrogate_name()
    test_model_cannot_replace_reliable_key_with_surrogate()
    test_model_cannot_reference_unknown_column()
    test_valid_detected_key_explanation()

    print()

    print(
        (
            "Dataset Identity Explanation version: "
            f"{DATASET_IDENTITY_EXPLANATION_RULE_VERSION}"
        )
    )

    print(
        "Dataset Identity Explanation v0.1: PASS"
    )


if __name__ == "__main__":
    main()
