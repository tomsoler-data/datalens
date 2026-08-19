from __future__ import annotations

import pandas as pd

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.preparation_output_explanation import (
    PREPARATION_OUTPUT_EXPLANATION_API_VERSION,
    router,
)

from app.preparation.analysis_output_explanation import (
    ANALYSIS_OUTPUT_EXPLANATION_RULE_VERSION,
    RawAnalysisOutputExplanation,
    build_analysis_output_recommendation_facts,
    build_output_explanation_llm_facts,
    terminal_dataset_ids,
    validate_analysis_output_explanation,
)

from app.preparation.preparation_artifact_store import (
    list_preparation_artifacts,
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)

from app.preparation.preparation_session import (
    create_preparation_session,
    reset_preparation_session_store_for_tests,
)


app = FastAPI()

app.include_router(
    router
)

client = TestClient(
    app
)


def reset_state() -> None:
    reset_preparation_session_store_for_tests()
    reset_preparation_artifact_store_for_tests()


def build_lineage():
    reset_state()

    session = create_preparation_session(
        selected_analysis_dataset_ids=[
            "orders",
            "customers",
            "products",
        ]
    )

    put_preparation_artifact(
        workflow_id=session.workflow_id,
        dataset_id="orders",
        dataset_filename="orders.csv",
        stage="clean",
        dataframe=pd.DataFrame(
            {
                "order_id": [
                    "O1",
                    "O2",
                ],
                "customer_id": [
                    "C1",
                    "C2",
                ],
                "product_id": [
                    "P1",
                    "P2",
                ],
            }
        ),
        parent_dataset_ids=[
            "orders",
        ],
        evidence_refs=[
            "cleaning:orders",
        ],
    )

    put_preparation_artifact(
        workflow_id=session.workflow_id,
        dataset_id="customers",
        dataset_filename="customers.csv",
        stage="clean",
        dataframe=pd.DataFrame(
            {
                "customer_id": [
                    "C1",
                    "C2",
                ],
                "segment": [
                    "A",
                    "B",
                ],
            }
        ),
        parent_dataset_ids=[
            "customers",
        ],
        evidence_refs=[
            "cleaning:customers",
        ],
    )

    put_preparation_artifact(
        workflow_id=session.workflow_id,
        dataset_id="products",
        dataset_filename="products.csv",
        stage="clean",
        dataframe=pd.DataFrame(
            {
                "product_id": [
                    "P1",
                    "P2",
                ],
                "category": [
                    "X",
                    "Y",
                ],
            }
        ),
        parent_dataset_ids=[
            "products",
        ],
        evidence_refs=[
            "cleaning:products",
        ],
    )

    put_preparation_artifact(
        workflow_id=session.workflow_id,
        dataset_id="orders_customers",
        dataset_filename="orders__customers.csv",
        stage="combine",
        dataframe=pd.DataFrame(
            {
                "order_id": [
                    "O1",
                    "O2",
                ],
                "customer_id": [
                    "C1",
                    "C2",
                ],
                "product_id": [
                    "P1",
                    "P2",
                ],
                "segment": [
                    "A",
                    "B",
                ],
            }
        ),
        parent_dataset_ids=[
            "orders",
            "customers",
        ],
        evidence_refs=[
            "join:orders_customers",
        ],
    )

    put_preparation_artifact(
        workflow_id=session.workflow_id,
        dataset_id="orders_customers_products",
        dataset_filename="orders__customers__products.csv",
        stage="combine",
        dataframe=pd.DataFrame(
            {
                "order_id": [
                    "O1",
                    "O2",
                ],
                "customer_id": [
                    "C1",
                    "C2",
                ],
                "product_id": [
                    "P1",
                    "P2",
                ],
                "segment": [
                    "A",
                    "B",
                ],
                "category": [
                    "X",
                    "Y",
                ],
            }
        ),
        parent_dataset_ids=[
            "orders_customers",
            "products",
        ],
        evidence_refs=[
            "join:orders_customers_products",
            "post_join_validation:passed",
        ],
    )

    return session


def test_terminal_frontier_ignores_self_parent() -> None:
    session = build_lineage()

    artifacts = list_preparation_artifacts(
        workflow_id=session.workflow_id
    )

    terminal = set(
        terminal_dataset_ids(
            artifacts
        )
    )

    assert terminal == {
        "orders_customers_products",
    }

    print(
        "Terminal lineage frontier ignores in-place self-parent: PASS"
    )


def test_final_combine_output_is_recommended() -> None:
    session = build_lineage()

    facts = build_analysis_output_recommendation_facts(
        workflow_id=session.workflow_id,
        dataset_id="orders_customers_products",
    )

    assert facts.is_terminal is True

    assert (
        facts.recommendation_status
        ==
        "recommended_terminal"
    )

    assert set(
        facts.root_dataset_ids
    ) == {
        "orders",
        "customers",
        "products",
    }

    assert (
        "orders_customers"
        in
        facts.ancestor_dataset_ids
    )

    assert facts.lineage_depth >= 2

    assert any(
        "consolidates 3 imported root datasets"
        in reason
        for reason in facts.deterministic_reasons
    )

    print(
        "Final COMBINE output receives deterministic recommendation: PASS"
    )


def test_intermediate_output_is_not_recommended() -> None:
    session = build_lineage()

    facts = build_analysis_output_recommendation_facts(
        workflow_id=session.workflow_id,
        dataset_id="orders_customers",
    )

    assert facts.is_terminal is False

    assert (
        facts.recommendation_status
        ==
        "superseded_intermediate"
    )

    print(
        "Intermediate lineage artifact is not recommended: PASS"
    )


def test_llm_receives_metadata_only() -> None:
    session = build_lineage()

    facts = build_analysis_output_recommendation_facts(
        workflow_id=session.workflow_id,
        dataset_id="orders_customers_products",
    )

    payload = build_output_explanation_llm_facts(
        facts
    )

    assert "rows" in payload
    assert "columns" in payload
    assert "root_dataset_filenames" in payload
    assert "dataframe" not in payload
    assert "sample_rows" not in payload
    assert "raw_values" not in payload

    print(
        "Output explanation sends metadata only to local model: PASS"
    )


def test_valid_model_explanation_is_accepted() -> None:
    session = build_lineage()

    facts = build_analysis_output_recommendation_facts(
        workflow_id=session.workflow_id,
        dataset_id="orders_customers_products",
    )

    raw = RawAnalysisOutputExplanation(
        confidence=0.97,
        title="Sortie finale recommandée",
        explanation=(
            "Cette sortie est terminale dans la lineage "
            "et regroupe les trois datasets racine."
        ),
        user_message=(
            "Elle évite d'analyser séparément des artefacts "
            "déjà représentés dans cette sortie combinée."
        ),
        referenced_dataset_ids=[
            "orders_customers_products",
            "orders",
            "customers",
            "products",
        ],
        cautions=[],
    )

    validated = validate_analysis_output_explanation(
        facts=facts,
        raw=raw,
    )

    assert validated.python_validated is True
    assert validated.executable is False

    assert (
        validated.recommendation_status
        ==
        "recommended_terminal"
    )

    print(
        "Valid local-model output explanation accepted: PASS"
    )


def test_model_cannot_reference_unknown_dataset() -> None:
    session = build_lineage()

    facts = build_analysis_output_recommendation_facts(
        workflow_id=session.workflow_id,
        dataset_id="orders_customers_products",
    )

    raw = RawAnalysisOutputExplanation(
        confidence=0.8,
        title="Sortie recommandée",
        explanation="Explication.",
        user_message="Message.",
        referenced_dataset_ids=[
            "invented_dataset",
        ],
        cautions=[],
    )

    try:
        validate_analysis_output_explanation(
            facts=facts,
            raw=raw,
        )

    except ValueError as error:
        assert (
            "unknown or unauthorized"
            in
            str(
                error
            )
        )

    else:
        raise AssertionError(
            "Unknown dataset reference should have been rejected."
        )

    print(
        "Invented dataset reference rejected: PASS"
    )


def test_api_without_ai() -> None:
    session = build_lineage()

    response = client.post(
        "/preparation/analysis-output/explain",
        json={
            "workflow_id":
                session.workflow_id,
            "dataset_id":
                "orders_customers_products",
            "include_ai":
                False,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body[
            "api_version"
        ]
        ==
        PREPARATION_OUTPUT_EXPLANATION_API_VERSION
    )

    assert body[
        "recommended"
    ] is True

    assert (
        body[
            "facts"
        ][
            "recommendation_status"
        ]
        ==
        "recommended_terminal"
    )

    assert body[
        "explanation"
    ] is None

    assert body[
        "ai_error"
    ] is None

    print(
        "Preparation output explanation API deterministic mode: PASS"
    )


def test_unknown_candidate_is_404() -> None:
    session = build_lineage()

    response = client.post(
        "/preparation/analysis-output/explain",
        json={
            "workflow_id":
                session.workflow_id,
            "dataset_id":
                "missing",
            "include_ai":
                False,
        },
    )

    assert response.status_code == 404

    print(
        "Unknown output candidate rejected: PASS"
    )


def test_route_registered() -> None:
    paths = app.openapi()[
        "paths"
    ]

    assert (
        "/preparation/analysis-output/explain"
        in paths
    )

    print(
        "Preparation output explanation route registered: PASS"
    )


def main() -> None:
    print(
        "=== DATALENS ANALYSIS OUTPUT EXPLANATION v0.1 ==="
    )

    print()

    test_terminal_frontier_ignores_self_parent()
    test_final_combine_output_is_recommended()
    test_intermediate_output_is_not_recommended()
    test_llm_receives_metadata_only()
    test_valid_model_explanation_is_accepted()
    test_model_cannot_reference_unknown_dataset()
    test_api_without_ai()
    test_unknown_candidate_is_404()
    test_route_registered()

    print()

    print(
        (
            "Analysis Output Explanation rule version: "
            f"{ANALYSIS_OUTPUT_EXPLANATION_RULE_VERSION}"
        )
    )

    print(
        "Analysis Output Explanation v0.1: PASS"
    )


if __name__ == "__main__":
    main()
