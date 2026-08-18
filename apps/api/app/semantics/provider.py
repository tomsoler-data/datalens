from __future__ import annotations

import json

from typing import (
    Any,
)

from app.ai.provider import (
    DEFAULT_MODEL,
    client,
)

from app.semantics.schemas import (
    ColumnSemanticDraft,
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SEMANTIC_COLUMN_SYSTEM_PROMPT = """
You are the semantic column profiler of DataLens.

Python has already inspected the structural properties
of the column.

Your task is to identify what the column represents.

You are NOT performing statistical analysis.

You are NOT ranking analyses.

You are NOT making business recommendations.

You are NOT inventing information that is absent from
the column name, dataset context, structural metadata or
sample values.

Return a semantic profile for exactly ONE column.

FIELDS

concept:
The specific underlying phenomenon represented by the
column.

Examples:
population
political_stability
drinking_water_access
customer_age
revenue
wash_mortality

domain:
The broader analytical domain.

Examples:
demography
governance
water_access
health
sales
customer
finance
geography
time

semantic_group:
A stable family shared by columns representing closely
related versions or measurements of the same underlying
phenomenon.

Examples:

basic drinking-water access
and
safely managed drinking-water access

may both use:
service_access

mortality rate
and
mortality count

may both use:
mortality

gross revenue
and
net revenue

may both use:
revenue

variant:
The meaningful version, level, subgroup or state that
distinguishes this column from another column in the
same concept or semantic group.

Examples:

basic drinking-water access:
basic

safely managed drinking-water access:
safely_managed

urban population:
urban

rural population:
rural

gross revenue:
gross

net revenue:
net

actual value:
actual

target value:
target

If no meaningful variant can be identified, use:
unknown

IMPORTANT:

Two columns may share the same concept and semantic_group
while having different variants.

For example:

concept:
drinking_water_access

semantic_group:
service_access

variant:
basic

versus:

concept:
drinking_water_access

semantic_group:
service_access

variant:
safely_managed

Use lowercase snake_case for:
- concept
- domain
- semantic_group
- variant
- qualifiers

Use "unknown" when the meaning cannot be inferred safely.

measure_kind must use only the allowed schema values.

unit_kind must use only the allowed schema values.

entity_role must use only the allowed schema values.

Do not contradict strong Python structural hints.

confidence:
high:
meaning is explicit from the column name and context

medium:
meaning is reasonably inferable but not fully explicit

low:
meaning is ambiguous

Return only the structured response requested by the
JSON schema.
""".strip()


# ============================================================
# RESPONSE EXTRACTION
# ============================================================

def extract_message_content(
    response: Any,
) -> str:
    message = getattr(
        response,
        "message",
        None,
    )


    if message is not None:
        content = getattr(
            message,
            "content",
            None,
        )


        if content is not None:
            return str(
                content
            )


    if isinstance(
        response,
        dict,
    ):
        message_data = response.get(
            "message",
            {}
        )


        if isinstance(
            message_data,
            dict,
        ):
            content = message_data.get(
                "content"
            )


            if content is not None:
                return str(
                    content
                )


    raise RuntimeError(
        (
            "Ollama returned a response without "
            "message content."
        )
    )


# ============================================================
# PROMPT
# ============================================================

def build_column_semantic_prompt(
    *,
    context: dict[
        str,
        Any,
    ],
    strict_retry: bool,
) -> str:
    retry_text = (
        """
The previous profile could not be validated.

Be conservative.

Use "unknown" rather than inventing meaning.

Respect the Python structural hints exactly when they
are not "unknown".
"""
        if strict_retry
        else ""
    )


    context_json = json.dumps(
        context,
        ensure_ascii=False,
        indent=2,
    )


    return f"""
Profile exactly the TARGET COLUMN.

{retry_text}

COLUMN CONTEXT:

{context_json}

Important:

- infer semantics from the target column;
- use the dataset and peer column names only as context;
- do not profile the other columns;
- reuse a broad semantic_group for closely related
  measures when appropriate;
- identify the variant when the column explicitly
  represents a level, subgroup or version;
- do not invent a business meaning that is unsupported;
- return only the required structured response.
""".strip()


# ============================================================
# MODEL CALL
# ============================================================

def call_column_semantic_model(
    *,
    context: dict[
        str,
        Any,
    ],
    model: str = DEFAULT_MODEL,
    strict_retry: bool = False,
) -> str:
    response = client.chat(
        model=
            model,

        messages=[
            {
                "role":
                    "system",

                "content":
                    SEMANTIC_COLUMN_SYSTEM_PROMPT,
            },
            {
                "role":
                    "user",

                "content":
                    build_column_semantic_prompt(
                        context=
                            context,

                        strict_retry=
                            strict_retry,
                    ),
            },
        ],

        format=(
            ColumnSemanticDraft
            .model_json_schema()
        ),

        options={
            "temperature":
                0.0,
        },
    )


    return extract_message_content(
        response
    )
