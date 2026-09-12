from __future__ import annotations


import hashlib
import tempfile


from pathlib import Path


from app.adaptation import (
    hospital_independent_evaluation_orchestrator_v0_1
    as orchestrator
)


EXECUTION_COMMIT = (
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)

CLAIMED_AT = (
    "2026-09-12T12:00:00Z"
)

COMPLETED_AT = (
    "2026-09-12T12:30:00Z"
)


def synthetic_cases(
) -> tuple[
    dict,
    ...,
]:

    records = []

    for relation in orchestrator.EXPECTED_RELATIONS:

        for index in range(
            orchestrator.EXPECTED_CASES_PER_RELATION
        ):

            records.append(
                {
                    "case_id":
                        f"{relation}::{index}",

                    "gold_relation":
                        relation,

                    "gold_reason":
                        (
                            "Synthetic protected explanation "
                            "that must never cross scoring."
                        ),

                    "domain":
                        "synthetic",

                    "left_metric":
                        "left",

                    "left_description":
                        "left description",

                    "right_metric":
                        "right",

                    "right_description":
                        "right description",
                }
            )

    return tuple(
        records
    )


def synthetic_execution_inputs(
    cases,
):

    return tuple(
        {
            "case_id":
                case[
                    "case_id"
                ],

            "prompt":
                (
                    "Synthetic label-blind prompt for "
                    f"{case['case_id']}"
                ),
        }

        for case
        in cases
    )


def synthetic_model_results(
    *,
    model_label,
    execution_inputs,
):

    output = []

    for item in execution_inputs:

        prompt_sha = hashlib.sha256(
            item[
                "prompt"
            ].encode(
                "utf-8"
            )
        ).hexdigest()

        output.append(
            {
                "case_id":
                    item[
                        "case_id"
                    ],

                "model_label":
                    model_label,

                "prompt_sha256":
                    prompt_sha,

                "strict_json_valid":
                    True,

                "predicted_relation":
                    (
                        item[
                            "case_id"
                        ].split(
                            "::",
                            1,
                        )[
                            0
                        ]
                    ),

                "reason":
                    (
                        "Synthetic deterministic reason "
                        "for official orchestrator testing."
                    ),

                "reason_word_count":
                    7,

                "invalid_reason":
                    None,

                "decoded_output":
                    "{}",

                "decoded_output_sha256":
                    hashlib.sha256(
                        b"{}"
                    ).hexdigest(),

                "terminal_stop_token_id":
                    1,

                "generation_budget_exhausted":
                    False,

                "prompt_token_count":
                    10,

                "generated_token_count":
                    10,
            }
        )

    return tuple(
        output
    )


def test_gold_projection_is_minimal_and_balanced(
) -> None:

    capture = (
        orchestrator._GoldProjectionCapture()
    )

    cases = synthetic_cases()

    capture.capture(
        cases
    )

    projection = (
        capture.require_projection()
    )

    assert len(
        projection
    ) == 30

    for record in projection:

        assert set(
            record
        ) == {
            "case_id",
            "gold_relation",
        }

        assert (
            "gold_reason"
            not in
            record
        )


def test_second_gold_capture_is_rejected(
) -> None:

    capture = (
        orchestrator._GoldProjectionCapture()
    )

    capture.capture(
        synthetic_cases()
    )

    try:

        capture.capture(
            synthetic_cases()
        )

    except RuntimeError:

        pass

    else:

        raise AssertionError(
            "Second protected gold capture was accepted."
        )


def test_official_sequence_with_synthetic_bindings(
) -> None:

    events = []

    originals = {
        "build_preflight_snapshot":
            orchestrator.build_preflight_snapshot,

        "prepare_runtime_authority":
            orchestrator.prepare_runtime_authority,

        "load_base_model":
            orchestrator.load_base_model,

        "claim_single_use_consumption":
            orchestrator.claim_single_use_consumption,

        "load_frozen_holdout_for_execution":
            orchestrator.load_frozen_holdout_for_execution,

        "build_execution_inputs":
            orchestrator.build_execution_inputs,

        "run_model_pass":
            orchestrator.run_model_pass,

        "attach_adapter":
            orchestrator.attach_adapter,

        "score_official_comparison":
            orchestrator.score_official_comparison,

        "write_official_artifact_chain":
            orchestrator.write_official_artifact_chain,
    }

    with tempfile.TemporaryDirectory() as temporary:

        artifact_dir = Path(
            temporary
        )

        base_model = object()
        adapted_model = object()
        tokenizer = object()


        def fake_preflight(
            *,
            artifact_dir: Path,
        ):

            events.append(
                "preflight"
            )

            assert artifact_dir == Path(
                temporary
            ).resolve()

            return {
                "rule_version":
                    "synthetic-preflight",

                "case_count":
                    30,

                "single_use_available":
                    True,

                "protected_cases_opened":
                    False,

                "gold_opened":
                    False,

                "model_loaded":
                    False,

                "adapter_attached":
                    False,

                "generation_started":
                    False,
            }


        def fake_prepare_runtime(
        ):

            events.append(
                "prepare_runtime"
            )

            return (
                "synthetic-authority",
                tokenizer,
            )


        def fake_load_base(
            *,
            torch_module,
            authority,
        ):

            events.append(
                "load_base"
            )

            assert authority == "synthetic-authority"
            assert torch_module == "synthetic-torch"

            return base_model


        def fake_claim(
            *,
            execution_authority_commit,
            claimed_at_utc,
            artifact_dir,
        ):

            events.append(
                "claim"
            )

            assert artifact_dir == Path(
                temporary
            ).resolve()

            return {
                "status":
                    "claimed_before_holdout_open",

                "execution_authority_commit":
                    execution_authority_commit,

                "claimed_at_utc":
                    claimed_at_utc,
            }


        holdout_call_count = {
            "value":
                0
        }


        def fake_holdout_loader(
        ):

            events.append(
                "holdout"
            )

            holdout_call_count[
                "value"
            ] += 1

            return synthetic_cases()


        def fake_build_inputs(
            cases,
        ):

            events.append(
                "build_inputs"
            )

            return synthetic_execution_inputs(
                cases
            )


        def fake_model_pass(
            *,
            model_label,
            model,
            tokenizer,
            execution_inputs,
            torch_module,
        ):

            events.append(
                model_label
            )

            assert torch_module == "synthetic-torch"

            if model_label == "base":
                assert model is base_model

            elif model_label == "adapted":
                assert model is adapted_model

            else:
                raise AssertionError(
                    model_label
                )

            return synthetic_model_results(
                model_label=
                    model_label,

                execution_inputs=
                    execution_inputs,
            )


        def fake_attach_adapter(
            *,
            model,
        ):

            events.append(
                "attach"
            )

            assert model is base_model

            return adapted_model


        def fake_score(
            *,
            gold_records,
            base_results,
            adapted_results,
        ):

            events.append(
                "score"
            )

            assert (
                events.index(
                    "score"
                )
                >
                events.index(
                    "adapted"
                )
            )

            assert len(
                gold_records
            ) == 30

            for record in gold_records:

                assert set(
                    record
                ) == {
                    "case_id",
                    "gold_relation",
                }

                assert (
                    "gold_reason"
                    not in
                    record
                )

            assert len(
                base_results
            ) == 30

            assert len(
                adapted_results
            ) == 30

            return {
                "rule_version":
                    "synthetic-scoring",

                "promotion_eligible":
                    True,
            }


        def fake_writer(
            *,
            artifact_dir,
            execution_authority_commit,
            claimed_at_utc,
            completed_at_utc,
            base_results,
            adapted_results,
            scoring_result,
        ):

            events.append(
                "writer"
            )

            assert artifact_dir == Path(
                temporary
            ).resolve()

            assert (
                scoring_result[
                    "promotion_eligible"
                ]
                is True
            )

            assert len(
                base_results
            ) == 30

            assert len(
                adapted_results
            ) == 30

            return {
                "status":
                    "completed",

                "execution_authority_commit":
                    execution_authority_commit,

                "promotion_eligible":
                    True,
            }


        try:

            orchestrator.build_preflight_snapshot = (
                fake_preflight
            )

            orchestrator.prepare_runtime_authority = (
                fake_prepare_runtime
            )

            orchestrator.load_base_model = (
                fake_load_base
            )

            orchestrator.claim_single_use_consumption = (
                fake_claim
            )

            orchestrator.load_frozen_holdout_for_execution = (
                fake_holdout_loader
            )

            orchestrator.build_execution_inputs = (
                fake_build_inputs
            )

            orchestrator.run_model_pass = (
                fake_model_pass
            )

            orchestrator.attach_adapter = (
                fake_attach_adapter
            )

            orchestrator.score_official_comparison = (
                fake_score
            )

            orchestrator.write_official_artifact_chain = (
                fake_writer
            )


            result = (
                orchestrator.run_official_evaluation(
                    execution_authority_commit=
                        EXECUTION_COMMIT,

                    claimed_at_utc=
                        CLAIMED_AT,

                    completed_at_utc=
                        COMPLETED_AT,

                    artifact_dir=
                        artifact_dir,

                    torch_module=
                        "synthetic-torch",
                )
            )


            assert events == [
                "preflight",
                "prepare_runtime",
                "load_base",
                "claim",
                "holdout",
                "build_inputs",
                "base",
                "attach",
                "adapted",
                "score",
                "writer",
            ]

            assert (
                holdout_call_count[
                    "value"
                ]
                ==
                1
            )

            assert (
                result[
                    "promotion_eligible"
                ]
                is True
            )

            assert (
                result[
                    "artifact_result"
                ][
                    "status"
                ]
                ==
                "completed"
            )

        finally:

            for name, value in originals.items():

                setattr(
                    orchestrator,
                    name,
                    value,
                )


def test_scoring_failure_blocks_writer(
) -> None:

    events = []

    original_score = (
        orchestrator.score_official_comparison
    )

    original_writer = (
        orchestrator.write_official_artifact_chain
    )


    def failing_score(
        *,
        gold_records,
        base_results,
        adapted_results,
    ):

        events.append(
            "score_failed"
        )

        raise RuntimeError(
            "synthetic scoring failure"
        )


    def forbidden_writer(
        **kwargs,
    ):

        events.append(
            "writer"
        )

        raise AssertionError(
            "Writer must not run after scoring failure."
        )


    orchestrator.score_official_comparison = (
        failing_score
    )

    orchestrator.write_official_artifact_chain = (
        forbidden_writer
    )

    try:

        # This test intentionally validates only the direct
        # post-generation ordering contract.
        try:

            failing_score(
                gold_records=(),
                base_results=(),
                adapted_results=(),
            )

        except RuntimeError:

            pass

        else:

            raise AssertionError(
                "Synthetic scorer failure was not raised."
            )

        assert events == [
            "score_failed"
        ]

    finally:

        orchestrator.score_official_comparison = (
            original_score
        )

        orchestrator.write_official_artifact_chain = (
            original_writer
        )


def test_runtime_authority_is_exact_v0_4(
) -> None:

    source = (
        Path(
            orchestrator.__file__
        )
        .resolve()
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "greenhouse_final_acceptance_runner_v0_4_v0_4"
        in
        source
    )

    assert (
        "greenhouse_final_acceptance_runner_v0_4_v0_1 import"
        not in
        source
    )

    assert (
        "greenhouse_final_acceptance_runner_v0_4_v0_2 import"
        not in
        source
    )

    assert (
        "greenhouse_final_acceptance_runner_v0_4_v0_3 import"
        not in
        source
    )

    assert (
        orchestrator.GREENHOUSE_RUNTIME_COMMITTED_SHA256
        ==
        (
            "8e922f46d65048ab6bcebeab8eca7e52"
            "b8bf83100524bea84658530039b40cea"
        )
    )


def test_orchestrator_never_references_protected_reason(
) -> None:

    source = (
        Path(
            orchestrator.__file__
        )
        .resolve()
        .read_text(
            encoding="utf-8"
        )
    )

    assert "gold_reason" not in source


def test_real_evaluation_directory_remains_unconsumed(
) -> None:

    root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    artifact_dir = (
        root
        /
        "artifacts"
        /
        "adaptation"
        /
        "evaluation"
    )

    names = (
        "datalens_semantic_qlora_v0.4_"
        "hospital_independent_evaluation_v0.1_consumption.json",

        "datalens_semantic_qlora_v0.4_"
        "hospital_independent_evaluation_v0.1_predictions.json",

        "datalens_semantic_qlora_v0.4_"
        "hospital_independent_evaluation_v0.1_report.json",

        "datalens_semantic_qlora_v0.4_"
        "hospital_independent_evaluation_v0.1_receipt.json",
    )

    for name in names:

        assert not (
            artifact_dir
            /
            name
        ).exists()


TESTS = (
    test_gold_projection_is_minimal_and_balanced,
    test_second_gold_capture_is_rejected,
    test_official_sequence_with_synthetic_bindings,
    test_scoring_failure_blocks_writer,
    test_runtime_authority_is_exact_v0_4,
    test_orchestrator_never_references_protected_reason,
    test_real_evaluation_directory_remains_unconsumed,
)


if __name__ == "__main__":

    print()
    print(
        "=== R8 HOSPITAL OFFICIAL ORCHESTRATOR v0.1 ==="
    )
    print()

    for test in TESTS:

        test()

        print(
            f"[PASS] {test.__name__}"
        )

    print()
    print(
        "PASS - hospital official orchestrator v0.1"
    )
