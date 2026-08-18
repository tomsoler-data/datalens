from __future__ import annotations

from pathlib import Path

from app.evals.ollama_baseline import (
    DEFAULT_BASELINE_MODEL,
    run_ollama_baseline,
    save_baseline_report,
)


BASE_DIR = Path(
    __file__,
).resolve().parent


BENCHMARK_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_reasoning_v1.jsonl"
)


RESULTS_DIR = (
    BASE_DIR
    / "evals"
    / "results"
)


def main() -> None:
    print(
        "=== DATALENS AI BASELINE v0.1 ==="
    )

    print()

    print(
        "Model:",
        DEFAULT_BASELINE_MODEL,
    )

    print(
        "Split: validation"
    )

    print()


    report = run_ollama_baseline(
        benchmark_path=
            BENCHMARK_PATH,

        split=
            "validation",

        model=
            DEFAULT_BASELINE_MODEL,
    )


    # ========================================================
    # BASIC INTEGRATION CONTRACT
    # ========================================================

    assert (
        report.case_count
        == 2
    )

    assert (
        report.split
        == "validation"
    )

    assert (
        report.model
        == DEFAULT_BASELINE_MODEL
    )


    result_case_ids = {
        result.case_id
        for result
        in report.results
    }


    # Frozen test case must NOT have been evaluated.
    assert (
        "ar_v1_006"
        not in result_case_ids
    )


    # Ollama + structured output must work for both
    # validation cases before this integration is considered
    # healthy.
    assert (
        report
        .generation_success_count
        == 2
    ), (
        "Au moins un cas n'a pas produit "
        "un AnalyticalCandidate valide."
    )


    assert (
        report
        .generation_error_count
        == 0
    )


    # ========================================================
    # CASE RESULTS
    # ========================================================

    for result in (
        report.results
    ):
        print(
            "=" * 72
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
            "Status:",
            result.status,
        )

        print(
            "Inference:",
            round(
                result.inference_ms,
                1,
            ),
            "ms",
        )


        if (
            result.candidate
            is not None
        ):
            candidate = (
                result.candidate
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
        "=" * 72
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 72
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
    # SAVE BASELINE
    # ========================================================

    output_path = (
        save_baseline_report(
            report=
                report,

            output_dir=
                RESULTS_DIR,
        )
    )


    print()

    print(
        "Saved:",
        output_path,
    )


    # ========================================================
    # IMPORTANT
    #
    # We intentionally assert NO minimum quality score.
    #
    # The purpose of this run is to measure the untouched
    # baseline, not to force Gemma to obtain a desired result.
    # ========================================================

    assert (
        0.0
        <= report.average_overall
        <= 1.0
    )


    print()

    print(
        "Gemma baseline captured: PASS"
    )


if __name__ == "__main__":
    main()