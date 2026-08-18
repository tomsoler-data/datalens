from __future__ import annotations

import json

from types import SimpleNamespace

from app.ai.dataset_dependency_extractor_v1 import (
    DATASET_DEPENDENCY_EXTRACTOR_VERSION,
    DATASET_DEPENDENCY_TEMPERATURE,
    DATASET_DEPENDENCY_THINKING,
    build_dependency_user_prompt,
    build_dependency_visible_context,
    dataset_dependency_extractor_metadata,
    extract_dataset_dependencies,
    require_dataset_dependencies,
)

from app.ai.dataset_dependency_prompt_v1 import (
    DATASET_DEPENDENCY_MODEL,
    DATASET_DEPENDENCY_PROMPT_VERSION,
    DATASET_DEPENDENCY_SYSTEM_PROMPT,
)

from app.planning.analytical_v1.dataset_context import (
    DatasetColumnSpec,
    DatasetContext,
)

from app.planning.analytical_v1.dependency import (
    DATASET_DEPENDENCY_CONTRACT_VERSION,
)

from app.planning.analytical_v1.relationships import (
    DatasetRelationshipSpec,
    RoutingRelationshipContext,
)


# ============================================================
# FAKE OLLAMA CLIENT
# ============================================================

class FakeChatClient:
    """
    Deterministic substitute for ollama.Client.
    """

    def __init__(
        self,
        content: str,
    ) -> None:

        self.content = (
            content
        )

        self.calls: list[
            dict
        ] = []


    def chat(
        self,
        **kwargs,
    ):

        self.calls.append(
            kwargs
        )


        return (
            SimpleNamespace(
                message=(
                    SimpleNamespace(
                        content=(
                            self.content
                        )
                    )
                )
            )
        )


# ============================================================
# FIXTURE
# ============================================================

def build_context() -> RoutingRelationshipContext:
    """
    Two semantic datasets connected by one trusted
    preparation-derived relationship.

    The relationship and available tools must NOT become
    visible to the dependency model.
    """

    sales = (
        DatasetContext(
            dataset_id=(
                "sales"
            ),

            filename=(
                "sales.csv"
            ),

            grain=(
                "order"
            ),

            entity_columns=[
                "customer_id",
            ],

            columns=[
                DatasetColumnSpec(
                    name=(
                        "order_id"
                    ),

                    analytical_type=(
                        "identifier"
                    ),

                    semantic_role=None,
                ),

                DatasetColumnSpec(
                    name=(
                        "customer_id"
                    ),

                    analytical_type=(
                        "identifier"
                    ),

                    semantic_role=None,
                ),

                DatasetColumnSpec(
                    name=(
                        "revenue"
                    ),

                    analytical_type=(
                        "quantitative"
                    ),

                    semantic_role=None,
                ),
            ],
        )
    )


    support = (
        DatasetContext(
            dataset_id=(
                "support"
            ),

            filename=(
                "support.csv"
            ),

            grain=(
                "ticket"
            ),

            entity_columns=[
                "customer_id",
            ],

            columns=[
                DatasetColumnSpec(
                    name=(
                        "ticket_id"
                    ),

                    analytical_type=(
                        "identifier"
                    ),

                    semantic_role=None,
                ),

                DatasetColumnSpec(
                    name=(
                        "customer_id"
                    ),

                    analytical_type=(
                        "identifier"
                    ),

                    semantic_role=None,
                ),

                DatasetColumnSpec(
                    name=(
                        "resolution_hours"
                    ),

                    analytical_type=(
                        "quantitative"
                    ),

                    semantic_role=None,
                ),
            ],
        )
    )


    relationship = (
        DatasetRelationshipSpec(
            relationship_id=(
                "sales_support_customer"
            ),

            left_dataset_id=(
                "sales"
            ),

            right_dataset_id=(
                "support"
            ),

            kind=(
                "join"
            ),

            left_keys=[
                "customer_id",
            ],

            right_keys=[
                "customer_id",
            ],

            validated=True,
        )
    )


    return (
        RoutingRelationshipContext(
            datasets=[
                sales,
                support,
            ],

            relationships=[
                relationship,
            ],

            available_tools=[
                "join_datasets",
                "aggregate",
                "measure_association",
            ],
        )
    )


# ============================================================
# 1. LOCKED CONFIGURATION
# ============================================================

def test_locked_configuration() -> None:

    assert (
        DATASET_DEPENDENCY_EXTRACTOR_VERSION
        == "dataset_dependency_extractor_v1.0"
    )


    assert (
        DATASET_DEPENDENCY_PROMPT_VERSION
        == "dataset_dependency_prompt_v0.8_baseline"
    )


    assert (
        DATASET_DEPENDENCY_MODEL
        == "qwen3:4b-instruct"
    )


    assert (
        DATASET_DEPENDENCY_TEMPERATURE
        == 0
    )


    assert (
        DATASET_DEPENDENCY_THINKING
        is False
    )


    assert (
        "Tu es le Dataset Dependency Extractor de DataLens."
        in DATASET_DEPENDENCY_SYSTEM_PROMPT
    )


    print(
        "Development-selected dependency configuration "
        "locked: PASS"
    )


# ============================================================
# 2. MODEL VISIBILITY BOUNDARY
# ============================================================

def test_visible_context_boundary() -> None:

    context = (
        build_context()
    )


    visible = (
        build_dependency_visible_context(
            user_request=(
                "Le revenu est-il associé au temps "
                "de résolution des tickets ?"
            ),

            context=(
                context
            ),
        )
    )


    assert (
        set(
            visible.keys()
        )
        == {
            "user_request",
            "datasets",
        }
    )


    serialized = (
        json.dumps(
            visible,
            ensure_ascii=False,
        )
    )


    assert (
        "sales_support_customer"
        not in serialized
    )


    assert (
        "join_datasets"
        not in serialized
    )


    assert (
        "measure_association"
        not in serialized
    )


    assert (
        "relationships"
        not in serialized
    )


    assert (
        "available_tools"
        not in serialized
    )


    assert (
        '"sales"'
        in serialized
    )


    assert (
        '"support"'
        in serialized
    )


    print(
        "Semantic-only model visibility boundary: PASS"
    )


# ============================================================
# 3. USER PROMPT
# ============================================================

def test_user_prompt_serialization() -> None:

    context = (
        build_context()
    )


    prompt = (
        build_dependency_user_prompt(
            user_request=(
                "Le revenu est-il associé au temps "
                "de résolution des tickets ?"
            ),

            context=(
                context
            ),
        )
    )


    assert (
        prompt.startswith(
            "CONTEXTE:\n\n"
        )
    )


    assert (
        "Identifie uniquement les groupes de datasets "
        "nécessaires aux résultats analytiques demandés."
        in prompt
    )


    assert (
        "sales_support_customer"
        not in prompt
    )


    assert (
        "join_datasets"
        not in prompt
    )


    print(
        "Historical dependency user prompt format "
        "preserved: PASS"
    )


# ============================================================
# 4. VALID SINGLE-DATASET CANDIDATE
# ============================================================

def test_valid_single_dataset_candidate() -> None:

    fake = (
        FakeChatClient(
            json.dumps(
                {
                    "requirements": [
                        {
                            "requirement_id":
                                "revenue_total",

                            "dataset_ids": [
                                "sales",
                            ],
                        }
                    ]
                }
            )
        )
    )


    result = (
        extract_dataset_dependencies(
            user_request=(
                "Quel est le chiffre d'affaires total ?"
            ),

            context=(
                build_context()
            ),

            chat_client=(
                fake
            ),
        )
    )


    assert (
        result.status
        == "valid"
    )


    assert (
        result.valid_for_feasibility_gate
    )


    assert (
        result.candidate
        is not None
    )


    assert (
        result.candidate
        .requirements[
            0
        ]
        .dataset_ids
        == [
            "sales",
        ]
    )


    print(
        "Valid single-dataset dependency candidate: PASS"
    )


# ============================================================
# 5. VALID CROSS-DATASET SEMANTIC CANDIDATE
# ============================================================

def test_valid_cross_dataset_candidate() -> None:

    fake = (
        FakeChatClient(
            json.dumps(
                {
                    "requirements": [
                        {
                            "requirement_id":
                                "revenue_vs_resolution",

                            "dataset_ids": [
                                "sales",
                                "support",
                            ],
                        }
                    ]
                }
            )
        )
    )


    result = (
        extract_dataset_dependencies(
            user_request=(
                "Le revenu est-il associé au temps "
                "de résolution des tickets ?"
            ),

            context=(
                build_context()
            ),

            chat_client=(
                fake
            ),
        )
    )


    assert (
        result.status
        == "valid"
    )


    assert (
        result.valid_for_feasibility_gate
    )


    assert (
        result.candidate
        is not None
    )


    assert (
        result.candidate
        .requirements[
            0
        ]
        .dataset_ids
        == [
            "sales",
            "support",
        ]
    )


    print(
        "Valid cross-dataset semantic candidate: PASS"
    )


# ============================================================
# 6. HALLUCINATED DATASET IS REJECTED
# ============================================================

def test_unknown_dataset_rejected() -> None:

    fake = (
        FakeChatClient(
            json.dumps(
                {
                    "requirements": [
                        {
                            "requirement_id":
                                "bad_requirement",

                            "dataset_ids": [
                                "sales",
                                "crm_secret",
                            ],
                        }
                    ]
                }
            )
        )
    )


    result = (
        extract_dataset_dependencies(
            user_request=(
                "Analyse les ventes avec le CRM."
            ),

            context=(
                build_context()
            ),

            chat_client=(
                fake
            ),
        )
    )


    assert (
        result.status
        == "invalid_candidate"
    )


    assert not (
        result.valid_for_feasibility_gate
    )


    assert (
        result.candidate
        is not None
    )


    assert (
        result.error
        is not None
    )


    assert (
        "crm_secret"
        in result.error
    )


    print(
        "Hallucinated dataset rejected by Python: PASS"
    )


# ============================================================
# 7. INVALID STRUCTURED OUTPUT
# ============================================================

def test_invalid_json_generation_error() -> None:

    fake = (
        FakeChatClient(
            "not-json"
        )
    )


    result = (
        extract_dataset_dependencies(
            user_request=(
                "Quel est le chiffre d'affaires total ?"
            ),

            context=(
                build_context()
            ),

            chat_client=(
                fake
            ),
        )
    )


    assert (
        result.status
        == "generation_error"
    )


    assert not (
        result.valid_for_feasibility_gate
    )


    assert (
        result.candidate
        is None
    )


    assert (
        result.error
        is not None
    )


    print(
        "Invalid structured output becomes "
        "generation_error: PASS"
    )


# ============================================================
# 8. EMPTY OUTPUT
# ============================================================

def test_empty_output_generation_error() -> None:

    fake = (
        FakeChatClient(
            ""
        )
    )


    result = (
        extract_dataset_dependencies(
            user_request=(
                "Quel est le chiffre d'affaires total ?"
            ),

            context=(
                build_context()
            ),

            chat_client=(
                fake
            ),
        )
    )


    assert (
        result.status
        == "generation_error"
    )


    assert (
        result.candidate
        is None
    )


    print(
        "Empty structured output becomes "
        "generation_error: PASS"
    )


# ============================================================
# 9. EXACT MODEL CALL CONFIGURATION
# ============================================================

def test_model_call_configuration() -> None:

    fake = (
        FakeChatClient(
            json.dumps(
                {
                    "requirements": [
                        {
                            "requirement_id":
                                "revenue_total",

                            "dataset_ids": [
                                "sales",
                            ],
                        }
                    ]
                }
            )
        )
    )


    context = (
        build_context()
    )


    extract_dataset_dependencies(
        user_request=(
            "Quel est le chiffre d'affaires total ?"
        ),

        context=(
            context
        ),

        chat_client=(
            fake
        ),
    )


    assert (
        len(
            fake.calls
        )
        == 1
    )


    call = (
        fake.calls[
            0
        ]
    )


    assert (
        call[
            "model"
        ]
        == "qwen3:4b-instruct"
    )


    assert (
        call[
            "options"
        ][
            "temperature"
        ]
        == 0
    )


    assert (
        call[
            "think"
        ]
        is False
    )


    assert (
        call[
            "format"
        ][
            "type"
        ]
        == "object"
    )


    assert (
        call[
            "messages"
        ][
            0
        ][
            "content"
        ]
        == DATASET_DEPENDENCY_SYSTEM_PROMPT
    )


    assert (
        "sales_support_customer"
        not in (
            call[
                "messages"
            ][
                1
            ][
                "content"
            ]
        )
    )


    print(
        "Historical model-call configuration preserved: PASS"
    )


# ============================================================
# 10. REQUIRE GUARD
# ============================================================

def test_require_guard_returns_valid_candidate() -> None:

    fake = (
        FakeChatClient(
            json.dumps(
                {
                    "requirements": [
                        {
                            "requirement_id":
                                "revenue_total",

                            "dataset_ids": [
                                "sales",
                            ],
                        }
                    ]
                }
            )
        )
    )


    candidate = (
        require_dataset_dependencies(
            user_request=(
                "Quel est le chiffre d'affaires total ?"
            ),

            context=(
                build_context()
            ),

            chat_client=(
                fake
            ),
        )
    )


    assert (
        candidate
        .requirements[
            0
        ]
        .dataset_ids
        == [
            "sales",
        ]
    )


    print(
        "Dependency guard returns only validated "
        "candidate: PASS"
    )


# ============================================================
# 11. REQUIRE GUARD REJECTS UNKNOWN DATASET
# ============================================================

def test_require_guard_rejects_unknown_dataset() -> None:

    fake = (
        FakeChatClient(
            json.dumps(
                {
                    "requirements": [
                        {
                            "requirement_id":
                                "bad_requirement",

                            "dataset_ids": [
                                "unknown_dataset",
                            ],
                        }
                    ]
                }
            )
        )
    )


    try:

        require_dataset_dependencies(
            user_request=(
                "Analyse le dataset inconnu."
            ),

            context=(
                build_context()
            ),

            chat_client=(
                fake
            ),
        )


    except ValueError as error:

        assert (
            "unknown_dataset"
            in str(
                error
            )
        )


        print(
            "Dependency guard rejects unknown dataset: PASS"
        )


    else:

        raise AssertionError(
            "Unknown dataset must never pass the "
            "dependency validation boundary."
        )


# ============================================================
# 12. METADATA
# ============================================================

def test_metadata() -> None:

    metadata = (
        dataset_dependency_extractor_metadata()
    )


    assert (
        metadata[
            "extractor_version"
        ]
        == DATASET_DEPENDENCY_EXTRACTOR_VERSION
    )


    assert (
        metadata[
            "prompt_version"
        ]
        == DATASET_DEPENDENCY_PROMPT_VERSION
    )


    assert (
        metadata[
            "contract_version"
        ]
        == DATASET_DEPENDENCY_CONTRACT_VERSION
    )


    assert (
        metadata[
            "model"
        ]
        == "qwen3:4b-instruct"
    )


    print(
        "Dependency extractor observability metadata: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS DATASET DEPENDENCY EXTRACTOR v1.0 ==="
    )


    print()


    test_locked_configuration()

    test_visible_context_boundary()

    test_user_prompt_serialization()

    test_valid_single_dataset_candidate()

    test_valid_cross_dataset_candidate()

    test_unknown_dataset_rejected()

    test_invalid_json_generation_error()

    test_empty_output_generation_error()

    test_model_call_configuration()

    test_require_guard_returns_valid_candidate()

    test_require_guard_rejects_unknown_dataset()

    test_metadata()


    print()


    print(
        "NO OLLAMA INFERENCE WAS PERFORMED."
    )


    print()


    print(
        "Dataset Dependency Extractor v1.0: PASS"
    )


if __name__ == "__main__":
    main()