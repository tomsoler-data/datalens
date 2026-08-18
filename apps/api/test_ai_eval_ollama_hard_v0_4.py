from __future__ import annotations

import json

from pathlib import Path


from app.evals.benchmark_loader import (
    load_benchmark,
)

from app.evals.ollama_baseline import (
    DEFAULT_BASELINE_MODEL,
)

from app.evals.ollama_baseline_v0_3 import (
    TYPED_CANDIDATE_CONTRACT_VERSION,
    run_typed_ollama_baseline,
)


BASE_DIR = Path(
    __file__,
).resolve().parent


BENCHMARK_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_reasoning_hard_v0_4.jsonl"
)


RESULTS_DIR = (
    BASE_DIR
    / "evals"
    / "results"
)


OUTPUT_PATH = (
    RESULTS_DIR
    / "gemma3_4b_validation_hard_v0_4.json"
)


def main() -> None:
    print(
        "=== DATALENS AI HARD EVAL v0.4 ==="
    )

    print()

    print(
        "Model:",
        DEFAULT_BASELINE_MODEL,
    )

    print(
        "Contract:",
        TYPED_CANDIDATE_CONTRACT_VERSION,
    )

    print(
        "Benchmark: hard validation v0.4"
    )

    print()


    # ========================================================
    # VERIFY BENCHMARK BEFORE CALLING THE LLM
    # ========================================================

    cases = load_benchmark(
        BENCHMARK_PATH,
        split="validation",
    )


    assert len(
        cases,
    ) == 6


    assert len(
        {
            case.case_id
            for case
            in cases
        }
    ) == 6


    assert all(
        case.split
        == "validation"
        for case
        in cases
    )


    assert all(
        case.expected.requires_reasoning
        for case
        in cases
    )


    print(
        "Benchmark contract: PASS"
    )

    print(
        "Validation cases:",
        len(
            cases,
        ),
    )

    print()


    # ========================================================
    # RUN EXACT SAME v0.3 MODEL CONTRACT
    # ========================================================

    report = (
        run_typed_ollama_baseline(
            benchmark_path=(
                BENCHMARK_PATH
            ),
            split="validation",
            model=(
                DEFAULT_BASELINE_MODEL
            ),
        )
    )


    assert (
        report.case_count
        == 6
    )


    assert (
        report.generation_success_count
        == 6
    ), (
        "Au moins un hard case n'a pas produit "
        "un TypedAnalyticalCandidate valide."
    )


    assert (
        report.generation_error_count
        == 0
    )


    # ========================================================
    # CASE RESULTS
    # ========================================================

    for result in report.results:
        print(
            "=" * 76
        )

        print(
            result.case_id,
            "|",
            result.domain,
        )

        print(
            "Question:",
            result.user_request,
        )

        print(
            "Inference:",
            round(
                result.inference_ms,
                1,
            ),
            "ms",
        )


        candidate = (
            result.candidate
        )


        assert (
            candidate is not None
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
            "Tool arguments:",
            [
                {
                    call.name:
                        call.arguments
                }

                for call
                in candidate.tool_calls
            ],
        )

        print(
            "Assumptions:",
            candidate.assumptions,
        )

        print(
            "Metrics:",
            {
                key:
                    round(
                        value,
                        3,
                    )

                for key, value
                in result
                .score_metrics
                .items()
            },
        )

        print(
            "Diagnostics:",
            result.score_diagnostics,
        )

        print(
            "Overall:",
            round(
                result.overall,
                3,
            ),
        )

        print()


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "=" * 76
    )

    print(
        "HARD BENCHMARK SUMMARY"
    )

    print(
        "=" * 76
    )


    print(
        "Cases:",
        report.case_count,
    )

    print(
        "Generation success:",
        report.generation_success_count,
    )

    print(
        "Generation errors:",
        report.generation_error_count,
    )

    print(
        "Average inference:",
        round(
            report.average_inference_ms,
            1,
        ),
        "ms",
    )

    print()


    for (
        metric_name,
        metric_value,
    ) in (
        report
        .average_metrics
        .items()
    ):
        print(
            f"{metric_name:<20}",
            round(
                metric_value,
                3,
            ),
        )


    print()

    print(
        "Average overall:",
        round(
            report.average_overall,
            3,
        ),
    )

    print(
        "Invented columns:",
        report.invented_column_count,
    )

    print(
        "Invented tools:",
        report.invented_tool_count,
    )

    print(
        "Forbidden tools:",
        report.forbidden_tool_count,
    )

    print(
        "Forbidden assumptions:",
        report.forbidden_assumption_count,
    )


    # ========================================================
    # CAPABILITY GROUPS
    # ========================================================

    comprehension = (
        (
            report.average_metrics[
                "intent"
            ]
            + report.average_metrics[
                "entity"
            ]
            + report.average_metrics[
                "grain"
            ]
            + report.average_metrics[
                "relevant_columns"
            ]
            + report.average_metrics[
                "family"
            ]
        )
        / 5
    )


    planning = (
        (
            report.average_metrics[
                "tool_selection"
            ]
            + report.average_metrics[
                "tool_arguments"
            ]
        )
        / 2
    )


    reliability = (
        report.average_metrics[
            "safety"
        ]
    )


    print()

    print(
        "=" * 76
    )

    print(
        "CAPABILITY VIEW"
    )

    print(
        "=" * 76
    )


    print(
        "Comprehension:",
        round(
            comprehension,
            3,
        ),
    )

    print(
        "Planning:",
        round(
            planning,
            3,
        ),
    )

    print(
        "Reliability:",
        round(
            reliability,
            3,
        ),
    )


    # ========================================================
    # SAVE WITHOUT OVERWRITING v0.1 / v0.2 / v0.3
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    payload = (
        report.model_dump(
            mode="json",
        )
    )


    payload[
        "evaluation_name"
    ] = (
        "hard_validation_v0.4"
    )


    payload[
        "capability_view"
    ] = {
        "comprehension":
            comprehension,

        "planning":
            planning,

        "reliability":
            reliability,
    }


    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print()

    print(
        "Saved:",
        OUTPUT_PATH,
    )


    # No quality threshold.
    #
    # A low score is useful evidence.
    assert (
        0.0
        <= report.average_overall
        <= 1.0
    )


    print()

    print(
        "Hard-model evaluation captured: PASS"
    )


if __name__ == "__main__":
    main()