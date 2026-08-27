from __future__ import annotations

import json

from typing import (
    Any,
)

from app.ai.provider import (
    DEFAULT_MODEL,
    client,
)

from app.security.llm_payload import (
    LLMPayloadClass,
    classified_llm_chat,
)

from app.ai.report_schemas import (
    SemanticCandidateAssessmentDraft,
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SEMANTIC_CANDIDATE_SYSTEM_PROMPT = """
You are the semantic assessment layer of DataLens.

Python already discovered, executed, validated and ranked
all analyses.

You evaluate exactly ONE target analysis at a time.

You do NOT select the final report.
Python will perform the final selection.

Your job is only to estimate how useful the target
analysis is from a semantic and analytical perspective.

The target is shown together with the other candidate
analyses so you can judge redundancy and complementarity.

STRICT RULES:

1. Evaluate only the TARGET candidate.

2. Do not calculate new statistics.

3. Do not invent facts.

4. Do not create business recommendations.

5. Do not claim causality.

6. semantic_relevance must be an integer from 0 to 100.

Use the full scale:

85-100:
very strong semantic relevance

70-84:
strong semantic relevance

50-69:
moderate semantic relevance

30-49:
limited semantic relevance

0-29:
low semantic relevance

7. Do NOT simply copy the deterministic analytical score.

The deterministic score measures analytical signal.
semantic_relevance measures how useful the finding is
to communicate and understand.

8. Prefer analyses that:
- reveal meaningful trends;
- reveal meaningful gaps;
- connect distinct concepts;
- connect distinct datasets;
- add a complementary perspective;
- reveal useful group differences.

9. Reduce semantic relevance when:
- two variables express nearly the same phenomenon;
- the relationship is structurally obvious;
- another candidate communicates essentially the same story;
- the result has little interpretive value despite a strong
  numerical signal.

10. semantic_priority must be:
- high
- medium
- low

11. reasons may contain only reason codes allowed by
the JSON schema.

12. Do not repeat the same reason code.

13. Return only the structured response requested by
the JSON schema.
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
        message_data = (
            response.get(
                "message",
                {}
            )
        )


        if isinstance(
            message_data,
            dict,
        ):
            content = (
                message_data.get(
                    "content"
                )
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

def build_semantic_candidate_prompt(
    *,
    target_key: str,
    target: dict[
        str,
        Any,
    ],
    candidate_catalog: list[
        dict[
            str,
            Any,
        ]
    ],
    objective: str | None,
    strict_retry: bool,
) -> str:
    objective_text = (
        objective
        or
        "No explicit objective. Auto Explore mode."
    )


    retry_instruction = (
        """
The previous assessment was invalid.

Return exactly one valid assessment.

Use semantic_relevance on the full 0-to-100 scale.
Do not repeat reason codes.
"""
        if strict_retry
        else ""
    )


    target_json = json.dumps(
        target,
        ensure_ascii=False,
        indent=2,
    )


    catalog_json = json.dumps(
        candidate_catalog,
        ensure_ascii=False,
        indent=2,
    )


    return f"""
Assess the semantic relevance of the TARGET analysis.

TARGET KEY:
{target_key}

USER OBJECTIVE:
{objective_text}

{retry_instruction}

TARGET ANALYSIS:

{target_json}

OTHER AVAILABLE CANDIDATES FOR COMPARISON:

{catalog_json}

Remember:

- assess only the target;
- do not select other candidates;
- semantic relevance is not the deterministic score;
- consider redundancy with other candidates;
- return only the required structured response.
""".strip()


# ============================================================
# PUBLIC MODEL CALL
# ============================================================

def call_semantic_candidate_model(
    *,
    target_key: str,
    target: dict[
        str,
        Any,
    ],
    candidate_catalog: list[
        dict[
            str,
            Any,
        ]
    ],
    objective: str | None,
    model: str = DEFAULT_MODEL,
    strict_retry: bool = False,
) -> str:
    response = classified_llm_chat(
        client,
        payload_class=(
            LLMPayloadClass
            .DETERMINISTIC_EVIDENCE
        ),
        model=
            model,

        messages=[
            {
                "role":
                    "system",

                "content":
                    SEMANTIC_CANDIDATE_SYSTEM_PROMPT,
            },
            {
                "role":
                    "user",

                "content":
                    build_semantic_candidate_prompt(
                        target_key=
                            target_key,

                        target=
                            target,

                        candidate_catalog=
                            candidate_catalog,

                        objective=
                            objective,

                        strict_retry=
                            strict_retry,
                    ),
            },
        ],

        format=(
            SemanticCandidateAssessmentDraft
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
