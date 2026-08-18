from __future__ import annotations

import json

from pathlib import Path

from app.evals.ollama_baseline import (
    DEFAULT_BASELINE_MODEL,
)

from app.evals.ollama_baseline_v0_2 import (
    OLLAMA_BASELINE_RULE_VERSION,
    STRICT_CANDIDATE_CONTRACT_VERSION,
    StrictAnalyticalCandidate,
    run_strict_ollama_baseline,
    save_strict_baseline_report,
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


BASELINE_V0_1_PATH = (
    RESULTS_DIR
    / "gemma3_4b_validation_baseline_v0_1.json"
)


def main() -> None:
    print(
        "=== DATALENS AI BASELINE v0.2 ==="
    )

    print()

    print(
        "Experiment:"
    )

    print(
        "Same model + same prompt + "
        "same benchmark + strict output contract"
    )

    print()

    print(
        "Model:",
        DEFAULT_BASELINE_MODEL,
    )

    print(
        "Split: validation"
    )

    print(
        "Rule:",
        OLLAMA_BASELINE_RULE_VERSION,
    )

    print(
        "Contract:",
        STRICT_CANDIDATE_CONTRACT_VERSION,
    )

    print()

    # ========================================================
    # VERIFY STRICT JSON SCHEMA
    # ========================================================

    schema = (
        StrictAnalyticalCandidate
        .model_json_schema()
    )

    required_fields = set(
        schema.get(
            "required",
            [],
        )
    )

    expected_required_fields = {
        "intent",
        "entity",
        "current_grain",
        "target_grain",
        "relevant_columns",
        "family",
        "tool_calls",
        "assumptions",
    }

    assert (
        required_fields
        == expected_required_fields
    ), (
        "Le contrat v0.2 n'impose pas "
        "exactement les champs attendus.\n"
        f"Required: {required_fields}"
    )

    print(
        "Strict schema required fields:",
        sorted(
            required_fields
        ),
    )

    print()

    # ========================================================
    # RUN
    # ========================================================

    report = (
        run_strict_ollama_baseline(
            benchmark_path=(
                BENCHMARK_PATH
            ),
            split="validation",
            model=(
                DEFAULT_BASELINE_MODEL
            ),
        )
    )

    assert report.case_count == 2

    assert report.split == "validation"

    assert (
        report.model
        == DEFAULT_BASELINE_MODEL
    )

    result_case_ids = {
        result.case_id
        for result
        in report.results
    }

    # Frozen test set remains untouched.
    assert (
        "ar_v1_006"
        not in result_case_ids
    )

    # Structured output itself should work.
    assert (
        report
        .generation_success_count
        == 2
    ), (
        "Au moins un cas n'a pas produit "
        "un StrictAnalyticalCandidate valide."
    )

    assert (
        report
        .generation_error_count
        == 0
    )

    # ========================================================
    # RESULTS
    # ========================================================

    for result in report.results:
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

        candidate = (
            result.candidate
        )

        assert (
            candidate is not None
        )

        # ----------------------------------------------------
        # Contract v0.2 should prevent these fields from being
        # omitted.
        # ----------------------------------------------------

        assert candidate.intent

        assert candidate.current_grain

        assert candidate.family

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
    # SUMMARY v0.2
    # ========================================================

    print(
        "=" * 72
    )

    print(
        "SUMMARY v0.2"
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
    # COMPARE AGAINST FROZEN v0.1 RESULT
    # ========================================================

    if BASELINE_V0_1_PATH.exists():
        baseline_v0_1 = json.loads(
            BASELINE_V0_1_PATH.read_text(
                encoding="utf-8",
            )
        )

        previous_score = float(
            baseline_v0_1[
                "average_overall"
            ]
        )

        delta = (
            report.average_overall
            - previous_score
        )

        print()
        print(
            "=" * 72
        )

        print(
            "COMPARISON"
        )

        print(
            "=" * 72
        )

        print(
            "v0.1 permissive:",
            round(
                previous_score,
                3,
            ),
        )

        print(
            "v0.2 strict:",
            round(
                report.average_overall,
                3,
            ),
        )

        print(
            "Delta:",
            (
                "+"
                if delta >= 0
                else ""
            )
            + str(
                round(
                    delta,
                    3,
                )
            ),
        )

    else:
        print()
        print(
            "WARNING:"
        )

        print(
            "Baseline v0.1 result file "
            "not found; comparison skipped."
        )

    # ========================================================
    # SAVE v0.2
    # ========================================================

    output_path = (
        save_strict_baseline_report(
            report=report,
            output_dir=RESULTS_DIR,
        )
    )

    print()
    print(
        "Saved:",
        output_path,
    )

    # ========================================================
    # NO QUALITY THRESHOLD
    #
    # We still do NOT require v0.2 to beat v0.1.
    # This is an experiment, not a target-score exercise.
    # ========================================================

    assert (
        0.0
        <= report.average_overall
        <= 1.0
    )

    print()
    print(
        "Strict-contract baseline captured: PASS"
    )


if __name__ == "__main__":
    main()