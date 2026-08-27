import json
import re
from typing import Any

from ollama import (
    Client,
)
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)

from app.ai.schemas import (
    AIFinding,
    AIWarning,
    DatasetAIExplanation,
    EvidenceClaim,
    EvidenceReference,
)

from app.security.llm_egress import (
    require_local_llm_url,
)

from app.security.llm_payload import (
    LLMPayloadClass,
    classified_llm_chat,
)


DEFAULT_MODEL = "gemma3:4b"

OLLAMA_HOST = (
    "http://localhost:11434"
)

MAX_FORMAT_ATTEMPTS = 2


client = Client(
    host=require_local_llm_url(
        OLLAMA_HOST
    ),
    follow_redirects=False,
    trust_env=False,
)


# ============================================================
# GENERIC DATALENS PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the AI explanation layer of DataLens.

Deterministic Python code is the source of truth.

You explain evidence.
You do NOT calculate results.

STRICT RULES:

1. Use only the provided evidence.

2. Never invent:
   - rows
   - values
   - percentages
   - statistics
   - operations
   - columns
   - transformations

3. Every factual finding must cite evidence.

4. Copy the exact evidence_id into "reference".

5. Copy the exact source_type.

6. Never invent evidence IDs.

7. Every finding must contain at least one
   machine-verifiable claim.

8. Claims must use fields that exist in the
   referenced evidence data.

9. Claim values must be copied exactly and
   represented as text.

Examples:

2 -> "2"
-1.0 -> "-1.0"
true -> "true"
Year -> "Year"

10. Never calculate, round, approximate or
    transform claim values.

11. If evidence exists, findings must not
    be empty.

12. Every cleaning_operation evidence item
    must be covered by a finding.

13. Do not put factual information only
    in the summary.

14. The summary may summarize findings,
    but must introduce no new facts.

15. normalize_missing_values means textual
    missing markers were converted to missing
    values. It does not mean rows were deleted.

16. Keep findings concise.

17. Do not repeat the same finding.

18. Return only the structured response
    requested by the JSON schema.
""".strip()


# ============================================================
# STATISTICAL NARRATIVE SCHEMA
# ============================================================

class StatisticalNarrativeFinding(
    BaseModel
):
    """
    Minimal natural-language output from
    the LLM.

    All factual numerical claims are added
    later by Python.
    """

    reference: str = Field(
        min_length=1,
    )

    statement: str = Field(
        min_length=1,
    )


class StatisticalNarrativeDraft(
    BaseModel
):
    """
    Lightweight statistical narrative.
    """

    findings: list[
        StatisticalNarrativeFinding
    ] = Field(
        min_length=1,
    )


# ============================================================
# STATISTICAL PROMPT
# ============================================================

STATISTICAL_SYSTEM_PROMPT = """
You are the statistical explanation layer
of DataLens.

Python and SciPy already performed all
calculations and interpretations required
to establish the factual result.

You do NOT calculate or infer statistics.

You receive explicit deterministic labels
such as:

- direction
- relationship_type
- significance

Your only job is to express those supplied
labels in concise natural language.

STRICT RULES:

1. Produce exactly one finding for each
   evidence item.

2. Copy the exact evidence_id into
   "reference".

3. Never invent an evidence ID.

4. Never omit an evidence ID.

5. Mention both variable names.

6. Mention the supplied statistical test.

7. Use the supplied relationship_type exactly
   in meaning.

8. Use the supplied direction exactly.

9. Use the supplied significance exactly.

10. Do NOT infer direction from a coefficient.

11. Do NOT infer significance from a p-value.

12. Do NOT include numerical values.

13. Do NOT include coefficients.

14. Do NOT include p-values.

15. Do NOT include alpha.

16. Do NOT include sample sizes.

17. Do NOT round or approximate anything.

18. Do NOT classify effect strength.

Never use words such as:

- strong
- weak
- moderate
- perfect
- small
- large

19. Never claim causation.

Never use causal wording such as:

- causes
- leads to
- drives
- results in

20. Correlation does not establish causation.

21. Keep every statement short.

Examples:

If the supplied evidence says:

test = Pearson
direction = negative
relationship_type = linear
significance = statistically significant

write something like:

"Pearson indicates a negative linear association
between Age and AverageBasket, and the result is
statistically significant."

If significance is:

not statistically significant

you must explicitly say:

"not statistically significant"

Return only the structured response requested
by the JSON schema.
""".strip()


# ============================================================
# GENERIC PROMPT BUILDER
# ============================================================

def build_user_prompt(
    context: dict[str, Any],
    strict_retry: bool = False,
) -> str:
    """
    Build the generic evidence prompt.

    Currently used primarily for cleaning.
    """

    evidence = context.get(
        "evidence",
        [],
    )

    required_ids = [
        str(
            item.get(
                "evidence_id"
            )
        )
        for item
        in evidence
        if (
            item.get(
                "evidence_id"
            )
            and item.get(
                "source_type"
            )
            == "cleaning_operation"
        )
    ]

    evidence_json = json.dumps(
        context,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
        default=str,
    )

    requirement = ""

    if required_ids:
        requirement = f"""
The following evidence IDs must be covered:

{json.dumps(
    required_ids,
    ensure_ascii=False,
)}

Do not return an empty findings list.

Every required evidence ID must appear in
at least one finding.
""".strip()

    retry_text = ""

    if strict_retry:
        retry_text = """
A previous response failed DataLens grounding
validation.

This is a strict retry.

Do not place evidence only in the summary.

Create concise evidence-backed findings.

Copy deterministic values exactly.
""".strip()

    return f"""
Explain the following DataLens evidence.

{requirement}

{retry_text}

EVIDENCE:

{evidence_json}
""".strip()


# ============================================================
# STATISTICAL DETERMINISTIC LABELS
# ============================================================

def get_statistical_direction(
    data: dict[str, Any],
) -> str:
    """
    Determine correlation direction in Python.

    The LLM never derives this from the
    coefficient.
    """

    coefficient = float(
        data[
            "coefficient"
        ]
    )

    if coefficient > 0:
        return "positive"

    if coefficient < 0:
        return "negative"

    return "zero"


def get_statistical_significance(
    data: dict[str, Any],
) -> str:
    """
    Convert the deterministic significance
    boolean into a narrative-ready label.
    """

    if bool(
        data.get(
            "statistically_significant"
        )
    ):
        return (
            "statistically significant"
        )

    return (
        "not statistically significant"
    )


# ============================================================
# SAFE STATISTICAL PROMPT CONTEXT
# ============================================================

def build_statistical_prompt_context(
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the context visible to the LLM.

    Numerical statistical results are
    deliberately excluded.

    The model receives only deterministic
    qualitative labels produced by Python.
    """

    safe_evidence = []

    for item in context.get(
        "evidence",
        [],
    ):
        if (
            item.get(
                "source_type"
            )
            != "statistical_result"
        ):
            continue

        data = item.get(
            "data",
            {},
        )

        safe_evidence.append(
            {
                "evidence_id":
                    str(
                        item[
                            "evidence_id"
                        ]
                    ),

                "source_type":
                    "statistical_result",

                "data": {
                    "x_column":
                        data.get(
                            "x_column"
                        ),

                    "y_column":
                        data.get(
                            "y_column"
                        ),

                    "test":
                        data.get(
                            "test"
                        ),

                    "relationship_type":
                        data.get(
                            "relationship_type"
                        ),

                    "direction":
                        get_statistical_direction(
                            data
                        ),

                    "significance":
                        get_statistical_significance(
                            data
                        ),
                },
            }
        )

    return {
        "dataset":
            context.get(
                "dataset"
            ),

        "evidence":
            safe_evidence,
    }


# ============================================================
# STATISTICAL PROMPT BUILDER
# ============================================================

def build_statistical_user_prompt(
    context: dict[str, Any],
    strict_retry: bool = False,
) -> str:
    """
    Ask the model to verbalize facts that
    Python has already classified.
    """

    safe_context = (
        build_statistical_prompt_context(
            context
        )
    )

    required_ids = [
        str(
            item[
                "evidence_id"
            ]
        )
        for item
        in safe_context[
            "evidence"
        ]
    ]

    evidence_json = json.dumps(
        safe_context,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
        default=str,
    )

    retry_text = ""

    if strict_retry:
        retry_text = """
A previous response was semantically invalid.

This is a strict retry.

Copy the supplied qualitative facts exactly.

In particular:

- use the supplied direction
- use the supplied relationship_type
- use the supplied significance

Do not infer anything yourself.
Do not include numbers.
""".strip()

    return f"""
Write exactly one short qualitative
interpretation for every evidence item.

Required evidence IDs:

{json.dumps(
    required_ids,
    ensure_ascii=False,
)}

{retry_text}

SAFE DETERMINISTIC EVIDENCE:

{evidence_json}
""".strip()


# ============================================================
# CONTEXT TYPE DETECTION
# ============================================================

def is_statistical_context(
    context: dict[str, Any],
) -> bool:
    """
    Return True when the evidence collection
    contains only statistical results.
    """

    evidence = context.get(
        "evidence",
        [],
    )

    if not evidence:
        return False

    return all(
        (
            item.get(
                "source_type"
            )
            == "statistical_result"
        )
        for item
        in evidence
    )


# ============================================================
# CLAIM VALUE CANONICALIZATION
# ============================================================

def canonicalize_statistical_claim_value(
    value: Any,
) -> str:
    """
    Convert deterministic Python values into
    exact textual claim representations.
    """

    if value is None:
        return "null"

    if isinstance(
        value,
        bool,
    ):
        return (
            "true"
            if value
            else "false"
        )

    if isinstance(
        value,
        float,
    ):
        return repr(
            value
        )

    return str(
        value
    )


STATISTICAL_CLAIM_FIELDS = (
    "x_column",
    "y_column",
    "n_total",
    "n_valid",
    "n_excluded",
    "test",
    "relationship_type",
    "coefficient_name",
    "coefficient",
    "p_value",
    "alternative",
    "n",
    "alpha",
    "statistically_significant",
)


# ============================================================
# GENERIC OLLAMA CALL
# ============================================================

def call_local_model(
    context: dict[str, Any],
    model: str,
    format_attempt: int,
    strict_retry: bool,
) -> str:
    """
    Call Ollama for generic DataLens evidence.
    """

    schema = (
        DatasetAIExplanation
        .model_json_schema()
    )

    response = (
        classified_llm_chat(
            client,
            payload_class=(
                LLMPayloadClass
                .DETERMINISTIC_EVIDENCE
            ),
            model=model,

            messages=[
                {
                    "role":
                        "system",

                    "content":
                        SYSTEM_PROMPT,
                },
                {
                    "role":
                        "user",

                    "content":
                        build_user_prompt(
                            context=
                                context,

                            strict_retry=
                                strict_retry,
                        ),
                },
            ],

            format=schema,

            stream=False,

            options={
                "temperature":
                    0,

                "num_ctx":
                    8192,

                "num_predict":
                    2200,

                "repeat_penalty":
                    1.15,

                "seed":
                    (
                        42
                        + format_attempt
                        + (
                            100
                            if strict_retry
                            else 0
                        )
                    ),
            },
        )
    )

    return (
        response
        .message
        .content
    )


# ============================================================
# STATISTICAL OLLAMA CALL
# ============================================================

def call_statistical_model(
    context: dict[str, Any],
    model: str,
    format_attempt: int,
    strict_retry: bool,
) -> str:
    """
    Ask the model only to verbalize the
    qualitative labels already determined
    by Python.
    """

    schema = (
        StatisticalNarrativeDraft
        .model_json_schema()
    )

    response = (
        classified_llm_chat(
            client,
            payload_class=(
                LLMPayloadClass
                .DETERMINISTIC_EVIDENCE
            ),
            model=model,

            messages=[
                {
                    "role":
                        "system",

                    "content":
                        STATISTICAL_SYSTEM_PROMPT,
                },
                {
                    "role":
                        "user",

                    "content":
                        build_statistical_user_prompt(
                            context=
                                context,

                            strict_retry=
                                strict_retry,
                        ),
                },
            ],

            format=schema,

            stream=False,

            options={
                "temperature":
                    0,

                "num_ctx":
                    8192,

                "num_predict":
                    600,

                "repeat_penalty":
                    1.15,

                "seed":
                    (
                        142
                        + format_attempt
                        + (
                            100
                            if strict_retry
                            else 0
                        )
                    ),
            },
        )
    )

    return (
        response
        .message
        .content
    )


# ============================================================
# STATEMENT PREPARATION
# ============================================================

def remove_column_names(
    statement: str,
    data: dict[str, Any],
) -> str:
    """
    Remove variable names before detecting
    forbidden numerical content.

    This allows names such as Revenue2026.
    """

    cleaned = statement

    for field in (
        "x_column",
        "y_column",
    ):
        value = data.get(
            field
        )

        if value is None:
            continue

        cleaned = re.sub(
            pattern=re.escape(
                str(
                    value
                )
            ),

            repl="",

            string=cleaned,

            flags=re.IGNORECASE,
        )

    return cleaned


# ============================================================
# STATEMENT SEMANTIC VALIDATION
# ============================================================

def validate_statistical_statement(
    reference: str,
    statement: str,
    data: dict[str, Any],
) -> None:
    """
    Validate the qualitative facts expressed
    by the model against Python evidence.
    """

    lowered = (
        statement
        .casefold()
    )

    x_column = str(
        data.get(
            "x_column",
            ""
        )
    )

    y_column = str(
        data.get(
            "y_column",
            ""
        )
    )

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    if (
        x_column.casefold()
        not in lowered
    ):
        raise ValueError(
            (
                f"{reference}: statement does "
                f"not mention x_column "
                f"{x_column!r}."
            )
        )

    if (
        y_column.casefold()
        not in lowered
    ):
        raise ValueError(
            (
                f"{reference}: statement does "
                f"not mention y_column "
                f"{y_column!r}."
            )
        )

    # --------------------------------------------------------
    # No numerical facts in LLM prose
    # --------------------------------------------------------

    statement_without_columns = (
        remove_column_names(
            statement=
                statement,

            data=
                data,
        )
    )

    if re.search(
        r"\d",
        statement_without_columns,
    ):
        raise ValueError(
            (
                f"{reference}: statistical "
                "statement contains a numerical "
                "value. Numerical facts must "
                "come from Python."
            )
        )

    # --------------------------------------------------------
    # Test name
    # --------------------------------------------------------

    test = str(
        data.get(
            "test",
            ""
        )
    ).casefold()

    if (
        test
        and test
        not in lowered
    ):
        raise ValueError(
            (
                f"{reference}: statement does "
                f"not mention test {test!r}."
            )
        )

    # --------------------------------------------------------
    # Relationship type
    # --------------------------------------------------------

    relationship_type = str(
        data.get(
            "relationship_type",
            ""
        )
    ).casefold()

    if (
        relationship_type
        and relationship_type
        not in lowered
    ):
        raise ValueError(
            (
                f"{reference}: statement does "
                "not contain relationship type "
                f"{relationship_type!r}."
            )
        )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    direction = (
        get_statistical_direction(
            data
        )
    )

    if direction == "positive":
        if (
            "positive"
            not in lowered
        ):
            raise ValueError(
                (
                    f"{reference}: expected "
                    "positive direction."
                )
            )

        if (
            "negative"
            in lowered
        ):
            raise ValueError(
                (
                    f"{reference}: contradictory "
                    "negative direction."
                )
            )

    elif direction == "negative":
        if (
            "negative"
            not in lowered
        ):
            raise ValueError(
                (
                    f"{reference}: expected "
                    "negative direction."
                )
            )

        if (
            "positive"
            in lowered
        ):
            raise ValueError(
                (
                    f"{reference}: contradictory "
                    "positive direction."
                )
            )

    else:
        if (
            "positive"
            in lowered
            or "negative"
            in lowered
        ):
            raise ValueError(
                (
                    f"{reference}: zero "
                    "coefficient must not be "
                    "described as positive "
                    "or negative."
                )
            )

    # --------------------------------------------------------
    # Statistical significance
    # --------------------------------------------------------

    is_significant = bool(
        data.get(
            "statistically_significant"
        )
    )

    not_significant_phrases = (
        "not statistically significant",
        "not significant",
    )

    if is_significant:
        if any(
            phrase in lowered
            for phrase
            in not_significant_phrases
        ):
            raise ValueError(
                (
                    f"{reference}: deterministic "
                    "result is significant, but "
                    "the statement says otherwise."
                )
            )

        if (
            "statistically significant"
            not in lowered
        ):
            raise ValueError(
                (
                    f"{reference}: statement "
                    "must say statistically "
                    "significant."
                )
            )

    else:
        if not any(
            phrase in lowered
            for phrase
            in not_significant_phrases
        ):
            raise ValueError(
                (
                    f"{reference}: statement "
                    "must explicitly say not "
                    "statistically significant."
                )
            )

    # --------------------------------------------------------
    # Unsupported strength labels
    # --------------------------------------------------------

    forbidden_strength_words = (
        "strong",
        "weak",
        "moderate",
        "perfect",
        "small",
        "large",
    )

    for word in (
        forbidden_strength_words
    ):
        if re.search(
            rf"\b{re.escape(word)}\b",
            lowered,
        ):
            raise ValueError(
                (
                    f"{reference}: unsupported "
                    "effect-strength description "
                    f"{word!r}."
                )
            )

    # --------------------------------------------------------
    # Causal wording
    # --------------------------------------------------------

    forbidden_causal_phrases = (
        "causes",
        "cause",
        "caused",
        "leads to",
        "lead to",
        "drives",
        "drive",
        "results in",
        "resulted in",
        "because of",
    )

    for phrase in (
        forbidden_causal_phrases
    ):
        if phrase in lowered:
            raise ValueError(
                (
                    f"{reference}: causal wording "
                    "is not allowed: "
                    f"{phrase!r}."
                )
            )


# ============================================================
# STATISTICAL DRAFT VALIDATION
# ============================================================

def validate_statistical_draft(
    draft: StatisticalNarrativeDraft,
    context: dict[str, Any],
) -> None:
    """
    Validate evidence coverage and qualitative
    statistical semantics.
    """

    evidence_index = {
        str(
            item[
                "evidence_id"
            ]
        ):
            item

        for item
        in context.get(
            "evidence",
            [],
        )

        if (
            item.get(
                "source_type"
            )
            == "statistical_result"
        )
    }

    required_ids = set(
        evidence_index.keys()
    )

    returned_ids = [
        finding.reference
        for finding
        in draft.findings
    ]

    if (
        len(
            returned_ids
        )
        != len(
            set(
                returned_ids
            )
        )
    ):
        raise ValueError(
            (
                "Statistical narrative contains "
                "duplicate evidence references."
            )
        )

    if (
        set(
            returned_ids
        )
        != required_ids
    ):
        raise ValueError(
            (
                "Statistical narrative does not "
                "cover exactly the required "
                "evidence IDs."
            )
        )

    for finding in (
        draft.findings
    ):
        evidence = (
            evidence_index[
                finding.reference
            ]
        )

        data = evidence.get(
            "data",
            {},
        )

        validate_statistical_statement(
            reference=
                finding.reference,

            statement=
                finding.statement,

            data=
                data,
        )


# ============================================================
# DETERMINISTIC FALLBACK NARRATIVE
# ============================================================

def build_deterministic_statistical_draft(
    context: dict[str, Any],
) -> StatisticalNarrativeDraft:
    """
    Build a safe narrative entirely in Python.

    Used only when the local LLM repeatedly
    fails semantic validation.
    """

    findings = []

    for evidence in context.get(
        "evidence",
        [],
    ):
        if (
            evidence.get(
                "source_type"
            )
            != "statistical_result"
        ):
            continue

        evidence_id = str(
            evidence[
                "evidence_id"
            ]
        )

        data = evidence.get(
            "data",
            {},
        )

        test = str(
            data.get(
                "test",
                "statistical test",
            )
        ).capitalize()

        x_column = str(
            data.get(
                "x_column",
                "x",
            )
        )

        y_column = str(
            data.get(
                "y_column",
                "y",
            )
        )

        relationship_type = str(
            data.get(
                "relationship_type",
                "association",
            )
        )

        direction = (
            get_statistical_direction(
                data
            )
        )

        significance = (
            get_statistical_significance(
                data
            )
        )

        if direction == "zero":
            statement = (
                f"{test} indicates a "
                f"{relationship_type} association "
                f"between {x_column} and "
                f"{y_column} with no directional "
                f"pattern, and the result is "
                f"{significance}."
            )

        else:
            statement = (
                f"{test} indicates a "
                f"{direction} "
                f"{relationship_type} association "
                f"between {x_column} and "
                f"{y_column}, and the result is "
                f"{significance}."
            )

        findings.append(
            StatisticalNarrativeFinding(
                reference=
                    evidence_id,

                statement=
                    statement,
            )
        )

    draft = (
        StatisticalNarrativeDraft(
            findings=
                findings,
        )
    )

    validate_statistical_draft(
        draft=draft,
        context=context,
    )

    return draft


# ============================================================
# DETERMINISTIC SUMMARY
# ============================================================

def build_statistical_summary(
    context: dict[str, Any],
) -> str:
    """
    Build the summary entirely in Python.
    """

    evidence = [
        item
        for item
        in context.get(
            "evidence",
            [],
        )
        if (
            item.get(
                "source_type"
            )
            == "statistical_result"
        )
    ]

    if not evidence:
        return (
            "No statistical evidence "
            "was available."
        )

    first_data = (
        evidence[
            0
        ].get(
            "data",
            {},
        )
    )

    x_column = str(
        first_data.get(
            "x_column",
            "x",
        )
    )

    y_column = str(
        first_data.get(
            "y_column",
            "y",
        )
    )

    tests = []

    for item in evidence:
        test = str(
            item.get(
                "data",
                {},
            ).get(
                "test",
                "",
            )
        )

        if (
            test
            and test
            not in tests
        ):
            tests.append(
                test
            )

    test_names = [
        test.capitalize()
        for test
        in tests
    ]

    if len(
        test_names
    ) == 1:
        tests_text = (
            test_names[
                0
            ]
        )

    elif len(
        test_names
    ) == 2:
        tests_text = (
            f"{test_names[0]} and "
            f"{test_names[1]}"
        )

    else:
        tests_text = ", ".join(
            test_names
        )

    return (
        "Statistical analysis of "
        f"{x_column} and {y_column} "
        "was completed using "
        f"{tests_text}. "
        "Exact numerical results are attached "
        "as validated deterministic evidence."
    )


# ============================================================
# DETERMINISTIC WARNINGS
# ============================================================

def build_statistical_warnings(
    context: dict[str, Any],
) -> list[
    AIWarning
]:
    """
    Surface deterministic statistical warnings
    with test-specific evidence attribution.
    """

    evidence_items = [
        item
        for item
        in context.get(
            "evidence",
            [],
        )
        if (
            item.get(
                "source_type"
            )
            == "statistical_result"
        )
    ]

    unique_messages = []

    for evidence in evidence_items:
        data = evidence.get(
            "data",
            {},
        )

        for warning in data.get(
            "warnings",
            [],
        ):
            message = str(
                warning
            )

            if (
                message
                not in unique_messages
            ):
                unique_messages.append(
                    message
                )

    warnings = []

    for message in (
        unique_messages
    ):
        lowered_message = (
            message.casefold()
        )

        relevant_references = []

        for evidence in evidence_items:
            evidence_id = str(
                evidence[
                    "evidence_id"
                ]
            )

            data = evidence.get(
                "data",
                {},
            )

            test = str(
                data.get(
                    "test",
                    "",
                )
            ).casefold()

            if (
                "spearman"
                in lowered_message
            ):
                if test == "spearman":
                    relevant_references.append(
                        evidence_id
                    )

                continue

            if (
                "pearson"
                in lowered_message
            ):
                if test == "pearson":
                    relevant_references.append(
                        evidence_id
                    )

                continue

            if (
                message
                in [
                    str(
                        value
                    )
                    for value
                    in data.get(
                        "warnings",
                        [],
                    )
                ]
            ):
                relevant_references.append(
                    evidence_id
                )

        warnings.append(
            AIWarning(
                message=
                    message,

                severity=
                    "info",

                evidence=[
                    EvidenceReference(
                        source_type=
                            "statistical_result",

                        reference=
                            reference,
                    )

                    for reference
                    in relevant_references
                ],
            )
        )

    return warnings


# ============================================================
# STATISTICAL EXPLANATION COMPOSITION
# ============================================================

def compose_statistical_explanation(
    draft: StatisticalNarrativeDraft,
    context: dict[str, Any],
) -> DatasetAIExplanation:
    """
    Combine validated natural language with
    deterministic Python claims.
    """

    narrative_by_reference = {
        finding.reference:
            finding.statement

        for finding
        in draft.findings
    }

    findings = []

    for evidence in context.get(
        "evidence",
        [],
    ):
        if (
            evidence.get(
                "source_type"
            )
            != "statistical_result"
        ):
            continue

        evidence_id = str(
            evidence[
                "evidence_id"
            ]
        )

        data = evidence.get(
            "data",
            {},
        )

        claims = []

        for field in (
            STATISTICAL_CLAIM_FIELDS
        ):
            if (
                field
                not in data
            ):
                continue

            claims.append(
                EvidenceClaim(
                    reference=
                        evidence_id,

                    field=
                        field,

                    value=
                        (
                            canonicalize_statistical_claim_value(
                                data[
                                    field
                                ]
                            )
                        ),
                )
            )

        findings.append(
            AIFinding(
                statement=
                    narrative_by_reference[
                        evidence_id
                    ],

                evidence=[
                    EvidenceReference(
                        source_type=
                            "statistical_result",

                        reference=
                            evidence_id,
                    )
                ],

                claims=
                    claims,
            )
        )

    dataset = str(
        context.get(
            "dataset",
            "dataset",
        )
    )

    return DatasetAIExplanation(
        dataset=
            dataset,

        summary=
            build_statistical_summary(
                context
            ),

        findings=
            findings,

        warnings=
            build_statistical_warnings(
                context
            ),
    )


# ============================================================
# STATISTICAL GENERATION
# ============================================================

def generate_statistical_explanation(
    context: dict[str, Any],
    model: str,
    strict_retry: bool,
) -> DatasetAIExplanation:
    """
    Generate a qualitative statistical
    explanation.

    The local LLM gets two attempts.

    If its language remains semantically
    invalid, DataLens uses a deterministic
    Python narrative instead.

    Numerical claims always come from Python.
    """

    errors = []

    for format_attempt in range(
        MAX_FORMAT_ATTEMPTS
    ):
        content = (
            call_statistical_model(
                context=
                    context,

                model=
                    model,

                format_attempt=
                    format_attempt,

                strict_retry=(
                    strict_retry
                    or format_attempt > 0
                ),
            )
        )

        try:
            draft = (
                StatisticalNarrativeDraft
                .model_validate_json(
                    content
                )
            )

            validate_statistical_draft(
                draft=
                    draft,

                context=
                    context,
            )

            return (
                compose_statistical_explanation(
                    draft=
                        draft,

                    context=
                        context,
                )
            )

        except (
            ValidationError,
            ValueError,
        ) as error:
            errors.append(
                str(
                    error
                )
            )

    # --------------------------------------------------------
    # Deterministic fallback
    # --------------------------------------------------------

    fallback_draft = (
        build_deterministic_statistical_draft(
            context
        )
    )

    return (
        compose_statistical_explanation(
            draft=
                fallback_draft,

            context=
                context,
        )
    )


# ============================================================
# GENERIC GENERATION
# ============================================================

def generate_generic_explanation(
    context: dict[str, Any],
    model: str,
    strict_retry: bool,
) -> DatasetAIExplanation:
    """
    Generic structured DataLens generation.

    Currently used for cleaning evidence.
    """

    errors = []

    for format_attempt in range(
        MAX_FORMAT_ATTEMPTS
    ):
        content = (
            call_local_model(
                context=
                    context,

                model=
                    model,

                format_attempt=
                    format_attempt,

                strict_retry=
                    strict_retry,
            )
        )

        try:
            return (
                DatasetAIExplanation
                .model_validate_json(
                    content
                )
            )

        except ValidationError as error:
            errors.append(
                {
                    "attempt":
                        format_attempt + 1,

                    "content_length":
                        len(
                            content
                        ),

                    "error":
                        str(
                            error
                        ),
                }
            )

    details = "; ".join(
        (
            f"attempt {item['attempt']}: "
            f"{item['content_length']} chars"
        )
        for item
        in errors
    )

    raise RuntimeError(
        (
            "The local LLM failed to return "
            "a valid DataLens structured response "
            f"after {MAX_FORMAT_ATTEMPTS} attempts. "
            f"{details}"
        )
    )


# ============================================================
# PUBLIC PROVIDER
# ============================================================

def generate_dataset_explanation(
    context: dict[str, Any],
    model: str = DEFAULT_MODEL,
    strict_retry: bool = False,
) -> DatasetAIExplanation:
    """
    Generate a DataLens explanation.

    Statistical pipeline:

        Python / SciPy facts
            ↓
        Python qualitative labels
            ↓
        LLM wording
            ↓
        semantic validation
            ↓
        deterministic fallback if needed
            ↓
        Python factual claims

    Generic evidence continues to use the
    standard structured LLM pipeline.
    """

    if is_statistical_context(
        context
    ):
        return (
            generate_statistical_explanation(
                context=
                    context,

                model=
                    model,

                strict_retry=
                    strict_retry,
            )
        )

    return (
        generate_generic_explanation(
            context=
                context,

            model=
                model,

            strict_retry=
                strict_retry,
        )
    )