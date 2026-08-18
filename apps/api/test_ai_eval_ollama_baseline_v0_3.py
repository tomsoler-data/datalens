from __future__ import annotations

import json

from pathlib import Path

from app.evals.ollama_baseline import (
    DEFAULT_BASELINE_MODEL,
)

from app.evals.ollama_baseline_v0_3 import (
    OLLAMA_BASELINE_RULE_VERSION,
    TYPED_CANDIDATE_CONTRACT_VERSION,
    TypedAnalyticalCandidate,
    run_typed_ollama_baseline,
    save_typed_baseline_report,
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


BASELINE_V0_2_PATH = (
    RESULTS_DIR
    / "gemma3_4b_validation_baseline_v0_2.json"
)


def read_previous_score(
    path: Path,
) -> float | None:
    if not path.exists():
        return None


    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


    return float(
        payload[
            "average_overall"
        ]
    )


def main() -> None:
    print(
        "=== DATALENS AI BASELINE v0.3 ==="
    )

    print()

    print(
        "Experiment:"
    )

    print(
        "Same model + same prompt + same benchmark + "
        "closed analytical vocabulary + typed tools"
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
        TYPED_CANDIDATE_CONTRACT_VERSION,
    )

    print()


    # ========================================================
    # VERIFY JSON SCHEMA
    # ========================================================

    schema = (
        TypedAnalyticalCandidate
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
    )


    intent_schema = (
        schema[
            "properties"
        ][
            "intent"
        ]
    )


    family_schema = (
        schema[
            "properties"
        ][
            "family"
        ]
    )


    print(
        "Required fields:",
        sorted(
            required_fields
        ),
    )

    print(
        "Intent schema:",
        intent_schema,
    )

    print(
        "Family schema:",
        family_schema,
    )

    print()


    # ========================================================
    # RUN
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


    assert (
        "ar_v1_006"
        not in result_case_ids
    )


    assert (
        report
        .generation_success_count
        == 2
    ), (
        "Au moins un cas n'a pas produit "
        "un TypedAnalyticalCandidate valide."
    )


    assert (
        report
        .generation_error_count
        == 0
    )


    # ========================================================
    # CASE RESULTS
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
        "SUMMARY v0.3"
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
    # HISTORY
    # ========================================================

    score_v0_1 = (
        read_previous_score(
            BASELINE_V0_1_PATH
        )
    )


    score_v0_2 = (
        read_previous_score(
            BASELINE_V0_2_PATH
        )
    )


    print()

    print(
        "=" * 72
    )

    print(
        "EXPERIMENT HISTORY"
    )

    print(
        "=" * 72
    )


    if (
        score_v0_1
        is not None
    ):
        print(
            "v0.1 permissive:",
            round(
                score_v0_1,
                3,
            ),
        )


    if (
        score_v0_2
        is not None
    ):
        print(
            "v0.2 strict:",
            round(
                score_v0_2,
                3,
            ),
        )


    print(
        "v0.3 typed:",
        round(
            report.average_overall,
            3,
        ),
    )


    if (
        score_v0_2
        is not None
    ):
        delta = (
            report.average_overall
            - score_v0_2
        )


        print(
            "Delta vs v0.2:",
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


    # ========================================================
    # SAVE
    # ========================================================

    output_path = (
        save_typed_baseline_report(
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
    # ========================================================

    assert (
        0.0
        <= report.average_overall
        <= 1.0
    )


    print()

    print(
        "Typed-contract baseline captured: PASS"
    )


if __name__ == "__main__":
    main()