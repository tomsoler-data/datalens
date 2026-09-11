from __future__ import annotations


import inspect


from pydantic import ValidationError


from app.adaptation.trusted_peft_smoke import (
    SMOKE_EOS_TOKEN_IDS,
    SMOKE_MAX_NEW_TOKENS,
    SMOKE_PAD_TOKEN_ID,
    TRUSTED_PEFT_SMOKE_RULE_VERSION,
    TrustedPEFTSmokeResult,
    run_trusted_peft_smoke,
    smoke_prompt_record,
)


def main(
) -> None:

    assert (
        TRUSTED_PEFT_SMOKE_RULE_VERSION
        ==
        "trusted_peft_smoke_v0.1"
    )


    assert (
        SMOKE_MAX_NEW_TOKENS
        ==
        64
    )


    assert (
        SMOKE_EOS_TOKEN_IDS
        ==
        (
            1,
            106,
        )
    )


    assert (
        SMOKE_PAD_TOKEN_ID
        ==
        0
    )


    prompt = (
        smoke_prompt_record()
    )


    assert set(
        prompt
    ) == {
        "domain",
        "left_metric",
        "left_description",
        "right_metric",
        "right_description",
    }


    serialized_prompt = repr(
        prompt
    ).casefold()


    for forbidden in (
        "greenhouse",
        "airport",
        "hotel",
        "holdout",
        "expected_relation",
        "target",
        "gold",
    ):

        assert forbidden not in serialized_prompt


    function_source = (
        inspect.getsource(
            run_trusted_peft_smoke
        )
    )


    resolver_index = (
        function_source.index(
            "resolve_trusted_peft_adapter_artifact("
        )
    )


    adaptation_runtime_import_index = (
        function_source.index(
            (
                "from app.adaptation."
                "qlora_runtime_v0_4 import"
            )
        )
    )


    canonicalizer_import_index = (
        function_source.index(
            (
                "from app.adaptation."
                "training_dataset_canonicalizer_v0_4 "
                "import"
            )
        )
    )


    torch_import_index = (
        function_source.index(
            "import torch"
        )
    )


    model_load_index = (
        function_source.index(
            ".from_pretrained(",
            function_source.index(
                "base_model = ("
            ),
        )
    )


    adapter_load_index = (
        function_source.index(
            ".from_pretrained(",
            function_source.index(
                "adapted_model = ("
            ),
        )
    )


    generate_index = (
        inspect.getsource(
            __import__(
                (
                    "app.adaptation."
                    "trusted_peft_smoke"
                ),
                fromlist=[
                    "_generate_once"
                ],
            )._generate_once
        )
        .index(
            ".generate("
        )
    )


    assert (
        resolver_index
        <
        adaptation_runtime_import_index
    )


    assert (
        resolver_index
        <
        canonicalizer_import_index
    )


    assert (
        adaptation_runtime_import_index
        <
        torch_import_index
    )


    assert (
        canonicalizer_import_index
        <
        torch_import_index
    )


    assert (
        torch_import_index
        <
        model_load_index
    )


    assert (
        model_load_index
        <
        adapter_load_index
    )


    assert (
        generate_index
        >=
        0
    )


    combined_source = (
        inspect.getsource(
            __import__(
                (
                    "app.adaptation."
                    "trusted_peft_smoke"
                ),
                fromlist=[
                    "*"
                ],
            )
        )
    )


    lowered = (
        combined_source.casefold()
    )


    for forbidden in (
        "greenhouse_final_acceptance",
        "airport_ground_operations_holdout",
        "snapshot_download",
        "hf_hub_download",
        "requests.get",
        ".backward(",
        "optimizer.step",
    ):

        assert forbidden not in lowered


    assert (
        "local_files_only="

        in
        combined_source
    )


    assert (
        "do_sample="

        in
        combined_source
    )


    assert (
        "num_beams="

        in
        combined_source
    )


    result = TrustedPEFTSmokeResult(
        artifact_id=
            "artifact:adapter:test",

        provenance_id=
            "provenance:llm_adaptation:test",

        experiment_id=
            "test",

        base_model_repository=
            "google/test",

        base_model_revision=
            "revision",

        base_model_checkpoint_path=
            "checkpoint",

        adapter_path=
            "adapter",

        prompt_sha256=
            "a" * 64,

        output_sha256=
            "b" * 64,

        generated_token_ids_sha256=
            "c" * 64,

        decoded_output=
            '{"relation":"uncertain"}',

        prompt_token_count=
            10,

        generated_token_count=
            5,

        deterministic_repeat_equal=
            True,

        cuda_device_name=
            "test-device",

        cuda_compute_capability=
            "8.9",

        runtime_versions={
            "torch":
                "test",
        },
    )


    try:
        result.decoded_output = "changed"

    except Exception:
        pass

    else:
        raise AssertionError(
            "Smoke result must be immutable."
        )


    try:

        TrustedPEFTSmokeResult(
            **{
                **result.model_dump(
                    mode="python"
                ),
                "generated_token_count":
                    65,
            }
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            "Generation budget must fail closed."
        )


    print(
        "Trusted resolver executes before load    PASS"
    )

    print(
        "Heavy runtime imports occur after trust  PASS"
    )

    print(
        "Synthetic prompt is label-blind          PASS"
    )

    print(
        "Protected-domain tokens absent           PASS"
    )

    print(
        "Local-only base-model loading            PASS"
    )

    print(
        "Deterministic generation contract        PASS"
    )

    print(
        "Training/backward surfaces absent        PASS"
    )

    print(
        "Immutable smoke result                   PASS"
    )

    print()
    print(
        "Trusted PEFT Smoke Contract v0.1: PASS"
    )


if __name__ == "__main__":
    main()
