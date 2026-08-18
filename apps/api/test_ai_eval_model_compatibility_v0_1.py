from __future__ import annotations

from time import perf_counter

from app.ai.provider import (
    client,
)

from app.evals.ollama_baseline import (
    SYSTEM_PROMPT,
)

from app.evals.ollama_baseline_v0_3 import (
    TypedAnalyticalCandidate,
)


# ============================================================
# VERSION
# ============================================================

COMPATIBILITY_TEST_VERSION = (
    "model_structured_output_compatibility_v0.1"
)


# ============================================================
# MODELS
# ============================================================

MODELS = [
    "qwen3:4b-instruct",
    "qwen3.5:4b",
    "phi4-mini",
    "ministral-3:3b",
]


# ============================================================
# SYNTHETIC CONTEXT
#
# This is deliberately NOT one of the benchmark cases.
#
# We only want to verify:
#
# model
#   -> Ollama
#   -> JSON Schema
#   -> TypedAnalyticalCandidate
#
# ============================================================

USER_PROMPT = """
CONTEXTE DISPONIBLE:

{
  "user_request": "Comment les ventes évoluent-elles au fil du temps ?",
  "datasets": [
    {
      "dataset_id": "sales",
      "filename": "sales.csv",
      "grain": "order_day",
      "entity_columns": [],
      "columns": [
        {
          "name": "date",
          "analytical_type": "temporal"
        },
        {
          "name": "revenue",
          "analytical_type": "quantitative"
        },
        {
          "name": "region",
          "analytical_type": "categorical"
        }
      ]
    }
  ],
  "available_tools": [
    {
      "name": "aggregate",
      "description": "Calcule des agrégations."
    },
    {
      "name": "analyze_time_series",
      "description": "Analyse l'évolution temporelle d'une variable."
    }
  ]
}

Construis le plus petit plan analytique permettant de répondre
directement à la demande.

Utilise uniquement les colonnes et outils fournis.
""".strip()


# ============================================================
# SINGLE MODEL
# ============================================================

def test_model(
    model: str,
) -> dict:
    print(
        "=" * 76
    )

    print(
        "Model:",
        model,
    )


    started_at = perf_counter()


    try:
        response = client.chat(
            model=model,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": USER_PROMPT,
                },
            ],

            format=(
                TypedAnalyticalCandidate
                .model_json_schema()
            ),

            options={
                "temperature": 0,
            },
        )


        inference_ms = (
            (
                perf_counter()
                - started_at
            )
            * 1000.0
        )


        raw_content = (
            response
            .message
            .content
        )


        candidate = (
            TypedAnalyticalCandidate
            .model_validate_json(
                raw_content,
            )
        )


        print(
            "Status: PASS"
        )

        print(
            "Inference:",
            round(
                inference_ms,
                1,
            ),
            "ms",
        )

        print(
            "Intent:",
            candidate.intent,
        )

        print(
            "Entity:",
            candidate.entity,
        )

        print(
            "Current grain:",
            candidate.current_grain,
        )

        print(
            "Target grain:",
            candidate.target_grain,
        )

        print(
            "Family:",
            candidate.family,
        )

        print(
            "Relevant columns:",
            candidate.relevant_columns,
        )

        print(
            "Tools:",
            [
                call.name
                for call
                in candidate.tool_calls
            ],
        )

        print(
            "Arguments:",
            [
                {
                    call.name:
                        call.arguments.model_dump(
                            mode="python",
                            exclude_none=True,
                        )
                }
                for call
                in candidate.tool_calls
            ],
        )

        print(
            "Assumptions:",
            candidate.assumptions,
        )


        # ----------------------------------------------------
        # Optional thinking channel
        #
        # Some Ollama reasoning models may expose reasoning
        # separately from message.content.
        #
        # It is informative but must NEVER be required for
        # parsing the DataLens structured output.
        # ----------------------------------------------------

        thinking = getattr(
            response.message,
            "thinking",
            None,
        )


        if thinking:
            print(
                "Separate thinking channel: YES"
            )

        else:
            print(
                "Separate thinking channel: NO"
            )


        return {
            "model":
                model,

            "status":
                "PASS",

            "inference_ms":
                inference_ms,

            "candidate":
                candidate,
        }


    except Exception as error:
        inference_ms = (
            (
                perf_counter()
                - started_at
            )
            * 1000.0
        )


        print(
            "Status: FAIL"
        )

        print(
            "Inference:",
            round(
                inference_ms,
                1,
            ),
            "ms",
        )

        print(
            "Error:",
            (
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )


        return {
            "model":
                model,

            "status":
                "FAIL",

            "inference_ms":
                inference_ms,

            "error":
                (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
        }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "=== DATALENS MODEL COMPATIBILITY v0.1 ==="
    )

    print(
        "Test:",
        COMPATIBILITY_TEST_VERSION,
    )

    print()

    print(
        "Purpose:"
    )

    print(
        "Verify Ollama structured-output compatibility "
        "with the DataLens typed analytical contract."
    )

    print()

    print(
        "Models:",
        len(
            MODELS,
        ),
    )

    print()


    results = [
        test_model(
            model,
        )
        for model
        in MODELS
    ]


    passed = [
        result
        for result
        in results
        if (
            result[
                "status"
            ]
            == "PASS"
        )
    ]


    failed = [
        result
        for result
        in results
        if (
            result[
                "status"
            ]
            == "FAIL"
        )
    ]


    print()

    print(
        "=" * 76
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 76
    )


    for result in results:
        print(
            f"{result['model']:<24}",
            result[
                "status"
            ],
            "|",
            round(
                result[
                    "inference_ms"
                ],
                1,
            ),
            "ms",
        )


    print()

    print(
        "Compatible:",
        len(
            passed,
        ),
        "/",
        len(
            results,
        ),
    )


    if failed:
        print()

        print(
            "Incompatible models:"
        )

        for result in failed:
            print(
                "-",
                result[
                    "model"
                ],
                ":",
                result[
                    "error"
                ],
            )


    print()

    print(
        "Compatibility test completed."
    )


if __name__ == "__main__":
    main()