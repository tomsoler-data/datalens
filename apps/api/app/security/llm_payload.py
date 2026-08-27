from __future__ import annotations


from enum import (
    Enum,
)

from typing import (
    Any,
)


# ============================================================
# VERSION
# ============================================================


LLM_PAYLOAD_PRIVACY_RULE_VERSION = (
    "llm_payload_privacy_v0.1"
)


# ============================================================
# PAYLOAD CLASS
# ============================================================


class LLMPayloadClass(
    str,
    Enum,
):
    """
    Classification of information intentionally made visible
    to a local DataLens model.

    METADATA_ONLY
        Dataset IDs, filenames, column names, grain, schemas,
        planner structure, tool names and similar metadata.

    DETERMINISTIC_EVIDENCE
        Facts, summaries, aggregates or decisions already
        produced by deterministic Python code.

    SEMANTIC_VALUE_SAMPLE
        Small, targeted value samples from one analytical
        field when semantic interpretation genuinely requires
        observing values.

        This class never authorizes complete tabular rows.

    DOCUMENT_CONTENT
        Uploaded documentary text intentionally processed by
        the local RAG / document understanding pipeline.

    TABULAR_RAW_ROWS
        Complete or row-shaped tabular records.

        This class is explicitly forbidden from model egress.
    """

    METADATA_ONLY = (
        "metadata_only"
    )

    DETERMINISTIC_EVIDENCE = (
        "deterministic_evidence"
    )

    SEMANTIC_VALUE_SAMPLE = (
        "semantic_value_sample"
    )

    DOCUMENT_CONTENT = (
        "document_content"
    )

    TABULAR_RAW_ROWS = (
        "tabular_raw_rows"
    )


# ============================================================
# ERROR
# ============================================================


class LLMPayloadPrivacyError(
    ValueError
):
    """
    Raised before model I/O when a payload class violates the
    DataLens model-visible privacy contract.
    """


# ============================================================
# ALLOWED CLASSES
# ============================================================


ALLOWED_LLM_PAYLOAD_CLASSES = frozenset(
    {
        LLMPayloadClass
        .METADATA_ONLY,

        LLMPayloadClass
        .DETERMINISTIC_EVIDENCE,

        LLMPayloadClass
        .SEMANTIC_VALUE_SAMPLE,

        LLMPayloadClass
        .DOCUMENT_CONTENT,
    }
)


# ============================================================
# VALIDATION
# ============================================================


def require_allowed_llm_payload_class(
    payload_class: (
        LLMPayloadClass
        |
        str
    ),
) -> LLMPayloadClass:
    """
    Validate the declared model-visible payload class.

    Raw tabular rows are fail-closed.

    Unknown or missing classifications are not silently
    converted to a permissive default.
    """

    if isinstance(
        payload_class,
        LLMPayloadClass,
    ):
        normalized = (
            payload_class
        )

    else:
        try:
            normalized = (
                LLMPayloadClass(
                    str(
                        payload_class
                    )
                    .strip()
                )
            )

        except (
            ValueError,
            TypeError,
        ) as error:
            raise LLMPayloadPrivacyError(
                (
                    "Unknown or invalid LLM "
                    "payload classification."
                )
            ) from error


    if (
        normalized
        ==
        LLMPayloadClass
        .TABULAR_RAW_ROWS
    ):
        raise LLMPayloadPrivacyError(
            (
                "Raw tabular rows are forbidden "
                "from DataLens model-visible "
                "payloads."
            )
        )


    if (
        normalized
        not in
        ALLOWED_LLM_PAYLOAD_CLASSES
    ):
        raise LLMPayloadPrivacyError(
            (
                "LLM payload classification is "
                "not authorized."
            )
        )


    return normalized


# ============================================================
# CLASSIFIED CHAT
# ============================================================


def classified_llm_chat(
    chat_client: Any,
    *,
    payload_class: (
        LLMPayloadClass
        |
        str
    ),
    **kwargs: Any,
) -> Any:
    """
    Execute one chat call only after an explicit privacy
    classification has passed the DataLens policy.

    payload_class is consumed locally and is never forwarded
    to the Ollama client.
    """

    require_allowed_llm_payload_class(
        payload_class
    )


    return (
        chat_client
        .chat(
            **kwargs
        )
    )


# ============================================================
# CLASSIFIED EMBEDDING
# ============================================================


def classified_llm_embed(
    embedding_client: Any,
    *,
    payload_class: (
        LLMPayloadClass
        |
        str
    ),
    **kwargs: Any,
) -> Any:
    """
    Execute one embedding call only after an explicit privacy
    classification has passed the DataLens policy.
    """

    require_allowed_llm_payload_class(
        payload_class
    )


    return (
        embedding_client
        .embed(
            **kwargs
        )
    )
