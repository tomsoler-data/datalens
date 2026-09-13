from __future__ import annotations


from typing import (
    Literal,
)


from app.ai.provider import (
    client,
)

from app.security.llm_payload import (
    LLMPayloadClass,
    classified_llm_chat,
)


# ============================================================
# VERSION
# ============================================================


SEMANTIC_OUTLIER_SCOPE_RULE_VERSION = (
    "semantic_outlier_scope_v0.1"
)


# ============================================================
# MODEL
# ============================================================


DEFAULT_SEMANTIC_OUTLIER_SCOPE_MODEL = (
    "qwen3.5:4b"
)


SEMANTIC_OUTLIER_SCOPE_NUM_CTX = 8192


# ============================================================
# LABELS
# ============================================================


SemanticOutlierScope = Literal[
    "customer_entity_outlier_detection",
    "variable_or_record_outlier_detection",
    "none",
]


_ALLOWED_SCOPE_LABELS = {
    "customer_entity_outlier_detection",
    "variable_or_record_outlier_detection",
    "none",
}


# ============================================================
# PROMPT
# ============================================================


SEMANTIC_OUTLIER_SCOPE_SYSTEM_PROMPT = """
Classify the analytical request by BOTH analytical concept
and analytical grain.

Choose exactly ONE label:

customer_entity_outlier_detection

Use when the requested analytical unit is the CUSTOMER and
the goal is to identify customer profiles whose behaviour is
unusually different, exceptional, anomalous, atypical, extreme,
or meaningfully distinct from the customer population.

Examples:
- customers whose purchasing behaviour differs strongly from
  the rest of the customer population;
- unusual customer profiles;
- anomalous customer behaviour.


variable_or_record_outlier_detection

Use when the anomaly request targets a VARIABLE VALUE,
OBSERVATION, ROW, RECORD, TRANSACTION, PRICE, MEASUREMENT,
or another non-customer analytical unit.

Examples:
- unusual prices;
- anomalous transactions;
- extreme measurements;
- outlier values in one quantitative variable.


none

Use for requests that are not asking for anomaly/outlier
detection.

Examples:
- Top-N customers by revenue;
- customer ranking;
- customer segmentation;
- distributions;
- descriptive statistics;
- comparisons between groups.

Important:
The word "customer" appearing somewhere in a request is not
sufficient for customer_entity_outlier_detection.
Determine what unit is actually supposed to be flagged.

Semantic equivalence matters.
Do not require exact keywords.

Output exactly one allowed label.
No numbering.
No explanation.
No JSON.
""".strip()


# ============================================================
# OUTPUT VALIDATION
# ============================================================


def parse_semantic_outlier_scope_label(
    raw: str,
) -> (
    SemanticOutlierScope
    | None
):
    """
    Validate one model-produced semantic-scope label.

    This function does not inspect or interpret the user's
    analytical objective. It validates only the model output.

    Any unexpected output fails closed.
    """

    normalized = str(
        raw
        or
        ""
    ).strip()

    if (
        normalized
        not in
        _ALLOWED_SCOPE_LABELS
    ):
        return None

    return normalized  # type: ignore[return-value]


# ============================================================
# MODEL RESOLUTION
# ============================================================


def resolve_semantic_outlier_scope(
    objective: str,
    *,
    model: str = (
        DEFAULT_SEMANTIC_OUTLIER_SCOPE_MODEL
    ),
) -> (
    SemanticOutlierScope
    | None
):
    """
    Resolve anomaly scope semantically with the local model.

    Only the user objective is supplied. No raw dataset row is
    required.

    Failure policy:
    - model/runtime failure -> None;
    - unexpected model output -> None.

    Therefore this resolver can never manufacture a customer
    entity-outlier route when classification is unavailable.
    """

    normalized_objective = str(
        objective
        or
        ""
    ).strip()

    if not normalized_objective:
        return None

    try:
        response = classified_llm_chat(
            client,
            payload_class=(
                LLMPayloadClass
                .METADATA_ONLY
            ),
            model=model,
            messages=[
                {
                    "role":
                        "system",
                    "content":
                        SEMANTIC_OUTLIER_SCOPE_SYSTEM_PROMPT,
                },
                {
                    "role":
                        "user",
                    "content":
                        normalized_objective,
                },
            ],
            think=False,
            stream=False,
            options={
                "temperature":
                    0,
                "seed":
                    42,
                "num_ctx":
                    SEMANTIC_OUTLIER_SCOPE_NUM_CTX,
            },
        )

    except Exception:
        return None

    raw = getattr(
        response.message,
        "content",
        "",
    )

    return (
        parse_semantic_outlier_scope_label(
            raw
        )
    )
