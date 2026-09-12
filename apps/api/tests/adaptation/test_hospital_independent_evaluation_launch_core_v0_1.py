from __future__ import annotations


import ast
import hashlib


from pathlib import Path


from app.adaptation import (
    hospital_independent_evaluation_launch_core_v0_1
    as launch_core
)

from app.adaptation import (
    hospital_independent_evaluation_preflight_v0_1
    as preflight
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MODULE_PATH = (
    ROOT
    /
    "app"
    /
    "adaptation"
    /
    "hospital_independent_evaluation_launch_core_v0_1.py"
)


def safe_snapshot(
) -> dict:

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


def synthetic_cases(
) -> tuple[
    dict,
    ...,
]:

    return tuple(
        {
            "case_id":
                f"hospital-{index:02d}",

            "domain":
                "hospital_emergency_department_operations",

            "metric_a_name":
                f"metric-a-{index}",

            "metric_a_description":
                "Synthetic metric A.",

            "metric_b_name":
                f"metric-b-{index}",

            "metric_b_description":
                "Synthetic metric B.",

            "gold_relation":
                "SENTINEL_DO_NOT_EXPOSE",

            "gold_reason":
                "SENTINEL_DO_NOT_EXPOSE",
        }

        for index
        in range(
            30
        )
    )


def build_dependencies(
    events: list,
    input_object_ids: list,
):

    authority = object()
    tokenizer = object()
    base_model = object()
    adapted_model = object()
    torch_sentinel = object()

    def build_preflight_snapshot():
        events.append(
            "preflight"
        )

        return safe_snapshot()

    def prepare_runtime_authority():
        events.append(
            "prepare_runtime"
        )

        return (
            authority,
            tokenizer,
        )

    def load_base_model(
        *,
        torch_module,
        authority,
    ):

        assert (
            events
            ==
            [
                "preflight",
                "prepare_runtime",
            ]
        )

        assert (
            torch_module
            is
            torch_sentinel
        )

        events.append(
            "load_base"
        )

        return base_model

    def claim_single_use_consumption(
        *,
        execution_authority_commit,
        claimed_at_utc,
    ):

        assert (
            events[
                -1
            ]
            ==
            "load_base"
        )

        events.append(
            "claim"
        )

        return {
            "status":
                "claimed_before_holdout_open",

            "execution_authority_commit":
                execution_authority_commit,

            "claimed_at_utc":
                claimed_at_utc,
        }

    def load_frozen_holdout_for_execution():

        assert (
            events[
                -1
            ]
            ==
            "claim"
        )

        events.append(
            "open_holdout"
        )

        return synthetic_cases()

    def build_execution_inputs(
        cases,
    ):

        assert (
            events[
                -1
            ]
            ==
            "open_holdout"
        )

        events.append(
            "build_inputs"
        )

        return tuple(
            {
                "case_id":
                    case[
                        "case_id"
                    ],

                "prompt":
                    (
                        "Synthetic label-blind prompt "
                        f"{case['case_id']}"
                    ),
            }

            for case
            in cases
        )

    def run_model_pass(
        *,
        model_label,
        model,
        tokenizer,
        execution_inputs,
        torch_module,
    ):

        assert (
            torch_module
            is
            torch_sentinel
        )

        assert all(
            set(
                item
            )
            ==
            {
                "case_id",
                "prompt",
            }

            for item
            in execution_inputs
        )

        assert all(
            "SENTINEL_DO_NOT_EXPOSE"
            not in
            item[
                "prompt"
            ]

            for item
            in execution_inputs
        )

        input_object_ids.append(
            id(
                execution_inputs
            )
        )

        if (
            model_label
            ==
            "base"
        ):

            assert (
                "attach_adapter"
                not in
                events
            )

            assert (
                model
                is
                base_model
            )

            events.append(
                "base_pass"
            )

        elif (
            model_label
            ==
            "adapted"
        ):

            assert (
                events[
                    -1
                ]
                ==
                "attach_adapter"
            )

            assert (
                model
                is
                adapted_model
            )

            events.append(
                "adapted_pass"
            )

        else:

            raise AssertionError(
                model_label
            )

        return tuple(
            {
                "case_id":
                    item[
                        "case_id"
                    ],

                "model_label":
                    model_label,

                "prompt_sha256":
                    hashlib.sha256(
                        item[
                            "prompt"
                        ]
                        .encode(
                            "utf-8"
                        )
                    ).hexdigest(),

                "strict_json_valid":
                    True,

                "relation":
                    "unrelated",

                "reason":
                    (
                        "Synthetic deterministic sentinel "
                        "reason for launch core testing."
                    ),
            }

            for item
            in execution_inputs
        )

    def attach_adapter(
        *,
        model,
    ):

        assert (
            events[
                -1
            ]
            ==
            "base_pass"
        )

        assert (
            model
            is
            base_model
        )

        events.append(
            "attach_adapter"
        )

        return adapted_model

    dependencies = {
        "build_preflight_snapshot":
            build_preflight_snapshot,

        "prepare_runtime_authority":
            prepare_runtime_authority,

        "load_base_model":
            load_base_model,

        "claim_single_use_consumption":
            claim_single_use_consumption,

        "load_frozen_holdout_for_execution":
            load_frozen_holdout_for_execution,

        "build_execution_inputs":
            build_execution_inputs,

        "run_model_pass":
            run_model_pass,

        "attach_adapter":
            attach_adapter,
    }

    return (
        dependencies,
        torch_sentinel,
    )


def test_frozen_authorities(
) -> None:

    assert (
        launch_core.RUNTIME_REUSE_FREEZE_PARENT_COMMIT
        ==
        "3472ffe6bcb4c2d435c7eabe70bc7a47e7b693ce"
    )

    assert (
        launch_core.GREENHOUSE_RUNTIME_COMMITTED_SHA256
        ==
        "8e922f46d65048ab6bcebeab8eca7e52b8bf83100524bea84658530039b40cea"
    )

    assert (
        launch_core.QLORA_RUNTIME_COMMITTED_SHA256
        ==
        "20e41ab00606296893276a84e53746c06618b8cabca74fef77cb743c5e80ab7c"
    )

    assert (
        launch_core.HOSPITAL_RUNNER_COMMITTED_SHA256
        ==
        "4f1f69ff01b9a9c096dd42fc694a6f9f11f0ba30a23f6fc2a8dddc92deff900d"
    )

    assert (
        launch_core.HOSPITAL_EXECUTION_COMMITTED_SHA256
        ==
        "28566b79ad34e2587ced709f68ca19b1dfaa5d4074c07ba769db3a11306ee1c1"
    )

    assert (
        launch_core.HOSPITAL_PREFLIGHT_COMMITTED_SHA256
        ==
        "bd9b55863b001e921c0cc398e2c9f7d7882eb111aa0fec07b231dfdcc5aca2a5"
    )


def test_exact_official_sequence_with_sentinels(
) -> None:

    events = []
    input_object_ids = []

    (
        dependencies,
        torch_module,
    ) = build_dependencies(
        events,
        input_object_ids,
    )

    result = (
        launch_core.run_official_launch_core(
            execution_authority_commit=
                (
                    "a"
                    *
                    40
                ),

            claimed_at_utc=
                "2026-09-12T14:00:00Z",

            dependencies=
                dependencies,

            torch_module=
                torch_module,
        )
    )

    assert (
        events
        ==
        [
            "preflight",
            "prepare_runtime",
            "load_base",
            "claim",
            "open_holdout",
            "build_inputs",
            "base_pass",
            "attach_adapter",
            "adapted_pass",
        ]
    )

    assert (
        len(
            input_object_ids
        )
        ==
        2
    )

    assert (
        input_object_ids[
            0
        ]
        ==
        input_object_ids[
            1
        ]
    )

    assert (
        result[
            "case_count"
        ]
        ==
        30
    )

    assert (
        result[
            "scoring_started"
        ]
        is False
    )

    assert (
        result[
            "report_written"
        ]
        is False
    )

    assert (
        result[
            "receipt_written"
        ]
        is False
    )


def test_unsafe_preflight_blocks_before_runtime(
) -> None:

    events = []

    def unsafe_preflight():

        events.append(
            "preflight"
        )

        snapshot = safe_snapshot()

        snapshot[
            "single_use_available"
        ] = False

        return snapshot

    def forbidden_dependency(
        *args,
        **kwargs,
    ):

        raise AssertionError(
            "Dependency executed after unsafe preflight."
        )

    dependencies = {
        key:
            (
                unsafe_preflight
                if
                key
                ==
                "build_preflight_snapshot"
                else
                forbidden_dependency
            )

        for key
        in launch_core.REQUIRED_DEPENDENCY_KEYS
    }

    try:

        launch_core.run_official_launch_core(
            execution_authority_commit=
                (
                    "b"
                    *
                    40
                ),

            claimed_at_utc=
                "2026-09-12T14:00:00Z",

            dependencies=
                dependencies,

            torch_module=
                object(),
        )

    except RuntimeError:

        pass

    else:

        raise AssertionError(
            "Unsafe preflight did not fail closed."
        )

    assert (
        events
        ==
        [
            "preflight"
        ]
    )


def test_base_load_failure_does_not_claim(
) -> None:

    events = []

    def build_preflight_snapshot():
        events.append(
            "preflight"
        )
        return safe_snapshot()

    def prepare_runtime_authority():
        events.append(
            "prepare_runtime"
        )
        return (
            object(),
            object(),
        )

    def load_base_model(
        *,
        torch_module,
        authority,
    ):
        events.append(
            "load_base"
        )
        raise RuntimeError(
            "synthetic base load failure"
        )

    def forbidden_dependency(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "Claim or protected read happened after failed base load."
        )

    dependencies = {
        "build_preflight_snapshot":
            build_preflight_snapshot,

        "prepare_runtime_authority":
            prepare_runtime_authority,

        "load_base_model":
            load_base_model,

        "claim_single_use_consumption":
            forbidden_dependency,

        "load_frozen_holdout_for_execution":
            forbidden_dependency,

        "build_execution_inputs":
            forbidden_dependency,

        "run_model_pass":
            forbidden_dependency,

        "attach_adapter":
            forbidden_dependency,
    }

    try:

        launch_core.run_official_launch_core(
            execution_authority_commit=
                (
                    "c"
                    *
                    40
                ),

            claimed_at_utc=
                "2026-09-12T14:00:00Z",

            dependencies=
                dependencies,

            torch_module=
                object(),
        )

    except RuntimeError as exc:

        assert (
            "synthetic base load failure"
            in
            str(
                exc
            )
        )

    else:

        raise AssertionError(
            "Synthetic base load failure was not propagated."
        )

    assert (
        events
        ==
        [
            "preflight",
            "prepare_runtime",
            "load_base",
        ]
    )


def test_execution_boundary_rejects_extra_field_before_generation(
) -> None:

    events = []
    input_object_ids = []

    (
        dependencies,
        torch_module,
    ) = build_dependencies(
        events,
        input_object_ids,
    )

    def bad_builder(
        cases,
    ):

        events.append(
            "build_inputs"
        )

        return tuple(
            {
                "case_id":
                    case[
                        "case_id"
                    ],

                "prompt":
                    "synthetic prompt",

                "forbidden_extra_field":
                    "must-not-cross",
            }

            for case
            in cases
        )

    dependencies[
        "build_execution_inputs"
    ] = bad_builder

    try:

        launch_core.run_official_launch_core(
            execution_authority_commit=
                (
                    "d"
                    *
                    40
                ),

            claimed_at_utc=
                "2026-09-12T14:00:00Z",

            dependencies=
                dependencies,

            torch_module=
                torch_module,
        )

    except RuntimeError:

        pass

    else:

        raise AssertionError(
            "Extra model-boundary field was accepted."
        )

    assert (
        "base_pass"
        not in
        events
    )

    assert (
        "attach_adapter"
        not in
        events
    )

    assert (
        "adapted_pass"
        not in
        events
    )


def test_base_pass_failure_blocks_adapter(
) -> None:

    events = []
    input_object_ids = []

    (
        dependencies,
        torch_module,
    ) = build_dependencies(
        events,
        input_object_ids,
    )

    original_run = dependencies[
        "run_model_pass"
    ]

    def failing_run(
        **kwargs,
    ):

        if (
            kwargs[
                "model_label"
            ]
            ==
            "base"
        ):
            events.append(
                "base_pass_failed"
            )
            raise RuntimeError(
                "synthetic base pass failure"
            )

        return original_run(
            **kwargs
        )

    dependencies[
        "run_model_pass"
    ] = failing_run

    try:

        launch_core.run_official_launch_core(
            execution_authority_commit=
                (
                    "e"
                    *
                    40
                ),

            claimed_at_utc=
                "2026-09-12T14:00:00Z",

            dependencies=
                dependencies,

            torch_module=
                torch_module,
        )

    except RuntimeError as exc:

        assert (
            "synthetic base pass failure"
            in
            str(
                exc
            )
        )

    else:

        raise AssertionError(
            "Synthetic Base-pass failure was not propagated."
        )

    assert (
        "attach_adapter"
        not in
        events
    )

    assert (
        "adapted_pass"
        not in
        events
    )


def test_module_has_no_direct_runtime_or_protected_authority(
) -> None:

    source = MODULE_PATH.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )

    imported_roots = set()

    for node in ast.walk(
        tree
    ):

        if isinstance(
            node,
            ast.Import,
        ):

            for alias in node.names:

                imported_roots.add(
                    alias.name.split(
                        "."
                    )[0]
                )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):

            if node.module:

                imported_roots.add(
                    node.module.split(
                        "."
                    )[0]
                )

    for forbidden in (
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
    ):

        assert (
            forbidden
            not in
            imported_roots
        )

    for forbidden_text in (
        "HOLDOUT_CASES_PATH",
        "gold_relation",
        "gold_reason",
        ".generate(",
        ".from_pretrained(",
        "os.open(",
        ".unlink(",
        ".write_text(",
        ".write_bytes(",
    ):

        assert (
            forbidden_text
            not in
            source
        )


def test_real_evaluation_directory_remains_unconsumed(
) -> None:

    preflight.assert_single_use_available()

    paths = (
        preflight.single_use_artifact_paths()
    )

    assert all(
        not path.exists()

        for path
        in paths.values()
    )


TESTS = (
    test_frozen_authorities,
    test_exact_official_sequence_with_sentinels,
    test_unsafe_preflight_blocks_before_runtime,
    test_base_load_failure_does_not_claim,
    test_execution_boundary_rejects_extra_field_before_generation,
    test_base_pass_failure_blocks_adapter,
    test_module_has_no_direct_runtime_or_protected_authority,
    test_real_evaluation_directory_remains_unconsumed,
)


def main(
) -> None:

    print(
        "=== R8 HOSPITAL OFFICIAL LAUNCH CORE v0.1 ==="
    )

    print()

    for test in TESTS:

        test()

        print(
            f"[PASS] {test.__name__}"
        )

    print()

    print(
        "PASS - hospital official launch core v0.1"
    )


if __name__ == "__main__":

    main()
