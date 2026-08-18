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


MODEL = "qwen3.5:4b"


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


def run_test(
    *,
    think: bool,
) -> None:
    print(
        "=" * 76
    )

    print(
        "Model:",
        MODEL,
    )

    print(
        "think:",
        think,
    )

    print()


    started_at = perf_counter()


    try:
        response = client.chat(
            model=MODEL,

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
                        USER_PROMPT,
                },
            ],

            format=(
                TypedAnalyticalCandidate
                .model_json_schema()
            ),

            options={
                "temperature":
                    0,
            },

            think=think,
        )


        elapsed_ms = (
            (
                perf_counter()
                - started_at
            )
            * 1000.0
        )


        content = (
            response
            .message
            .content
            or ""
        )


        thinking = (
            getattr(
                response.message,
                "thinking",
                None,
            )
            or ""
        )


        print(
            "Inference:",
            round(
                elapsed_ms,
                1,
            ),
            "ms",
        )

        print(
            "Content length:",
            len(
                content,
            ),
        )

        print(
            "Thinking length:",
            len(
                thinking,
            ),
        )

        print()


        if thinking:
            preview = (
                thinking[
                    :500
                ]
            )

            print(
                "Thinking preview:"
            )

            print(
                preview
            )

            print()


        if not content:
            print(
                "Structured content: EMPTY"
            )

            print()

            return


        print(
            "Raw structured content:"
        )

        print(
            content
        )

        print()


        candidate = (
            TypedAnalyticalCandidate
            .model_validate_json(
                content,
            )
        )


        print(
            "Structured parsing: PASS"
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
                        call
                        .arguments
                        .model_dump(
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


    except Exception as error:
        elapsed_ms = (
            (
                perf_counter()
                - started_at
            )
            * 1000.0
        )


        print(
            "Inference:",
            round(
                elapsed_ms,
                1,
            ),
            "ms",
        )

        print(
            "Status: FAIL"
        )

        print(
            "Error:",
            (
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )


def main() -> None:
    print(
        "=== DATALENS QWEN 3.5 THINKING TEST v0.1 ==="
    )

    print()

    print(
        "Test 1 — thinking disabled"
    )

    print()

    run_test(
        think=False,
    )


    print()

    print(
        "Test completed."
    )


if __name__ == "__main__":
    main()