from __future__ import annotations


import ast
import json


from pathlib import Path
from tempfile import TemporaryDirectory


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
    "hospital_independent_evaluation_preflight_v0_1.py"
)


def require_raises(
    exception_type,
    function,
) -> None:

    try:

        function()

    except exception_type:

        return

    raise AssertionError(
        (
            f"Expected "
            f"{exception_type.__name__}"
        )
    )


def test_frozen_authority_constants(
) -> None:

    assert (
        preflight.D4_C_B_FOUNDATION_COMMIT
        ==
        "67bf0bccaabf63afc42a832e56ebea47ca559acd"
    )

    assert (
        preflight.BASE_MODEL_REPOSITORY
        ==
        "google/gemma-3-4b-it"
    )

    assert (
        preflight.BASE_MODEL_REVISION
        ==
        "093f9f388b31de276ce2de164bdc2081324b9767"
    )

    assert (
        preflight.EXPECTED_CASE_COUNT
        ==
        30
    )


def test_committed_source_authority_is_platform_independent(
) -> None:

    qlora_path = (
        "apps/api/app/adaptation/"
        "qlora_runtime_v0_4.py"
    )

    expected = (
        "20e41ab00606296893276a84e53746c0"
        "6618b8cabca74fef77cb743c5e80ab7c"
    )

    assert (
        preflight.SOURCE_AUTHORITIES[
            qlora_path
        ]
        ==
        expected
    )

    assert (
        preflight.git_committed_sha256(
            repo_relative_path=
                qlora_path,
        )
        ==
        expected
    )


def test_source_eol_normalization(
) -> None:

    lf = (
        b"alpha\n"
        b"beta\n"
        b"gamma\n"
    )

    crlf = (
        b"alpha\r\n"
        b"beta\r\n"
        b"gamma\r\n"
    )

    cr = (
        b"alpha\r"
        b"beta\r"
        b"gamma\r"
    )

    assert (
        preflight.normalize_source_eol(
            lf
        )
        ==
        lf
    )

    assert (
        preflight.normalize_source_eol(
            crlf
        )
        ==
        lf
    )

    assert (
        preflight.normalize_source_eol(
            cr
        )
        ==
        lf
    )


def test_source_authority_rejects_non_eol_drift_contract(
) -> None:

    committed = (
        b"alpha\n"
        b"beta\n"
    )

    eol_only_working = (
        b"alpha\r\n"
        b"beta\r\n"
    )

    changed_working = (
        b"alpha\r\n"
        b"CHANGED\r\n"
    )

    assert (
        preflight.normalize_source_eol(
            eol_only_working
        )
        ==
        preflight.normalize_source_eol(
            committed
        )
    )

    assert (
        preflight.normalize_source_eol(
            changed_working
        )
        !=
        preflight.normalize_source_eol(
            committed
        )
    )


def test_real_preflight_snapshot_without_consumption(
) -> None:

    snapshot = (
        preflight.build_preflight_snapshot()
    )

    assert (
        snapshot[
            "single_use_available"
        ]
        is True
    )

    assert (
        snapshot[
            "protected_cases_opened"
        ]
        is False
    )

    assert (
        snapshot[
            "gold_opened"
        ]
        is False
    )

    assert (
        snapshot[
            "model_loaded"
        ]
        is False
    )

    assert (
        snapshot[
            "generation_started"
        ]
        is False
    )

    assert (
        snapshot[
            "case_count"
        ]
        ==
        30
    )


def test_single_use_availability_blocks_every_terminal_artifact(
) -> None:

    with TemporaryDirectory() as directory:

        root = Path(
            directory
        )

        paths = (
            preflight.single_use_artifact_paths(
                artifact_dir=
                    root
            )
        )

        for (
            label,
            path,
        ) in paths.items():

            for cleanup in paths.values():

                if cleanup.exists():

                    cleanup.unlink()

            path.write_text(
                "{}\n",
                encoding="utf-8",
            )

            require_raises(
                RuntimeError,
                lambda:
                    preflight.assert_single_use_available(
                        artifact_dir=
                            root
                    ),
            )

            assert (
                path.exists()
            ), label


def test_atomic_single_use_claim(
) -> None:

    with TemporaryDirectory() as directory:

        root = Path(
            directory
        )

        payload = (
            preflight.claim_single_use_consumption(
                execution_authority_commit=
                    (
                        "a"
                        *
                        40
                    ),

                claimed_at_utc=
                    "2026-09-12T10:30:00Z",

                artifact_dir=
                    root,
            )
        )

        paths = (
            preflight.single_use_artifact_paths(
                artifact_dir=
                    root
            )
        )

        marker = (
            paths[
                "consumption_marker"
            ]
        )

        assert marker.is_file()

        stored = json.loads(
            marker.read_text(
                encoding="utf-8"
            )
        )

        assert stored == payload

        assert (
            stored[
                "status"
            ]
            ==
            "claimed_before_holdout_open"
        )

        assert (
            stored[
                "single_use"
            ]
            is True
        )

        assert (
            stored[
                "holdout_opened_before_claim"
            ]
            is False
        )

        assert (
            stored[
                "gold_opened_before_claim"
            ]
            is False
        )


def test_second_claim_fails_closed(
) -> None:

    with TemporaryDirectory() as directory:

        root = Path(
            directory
        )

        preflight.claim_single_use_consumption(
            execution_authority_commit=
                (
                    "b"
                    *
                    40
                ),

            claimed_at_utc=
                "2026-09-12T10:30:00Z",

            artifact_dir=
                root,
        )

        marker = (
            preflight.single_use_artifact_paths(
                artifact_dir=
                    root
            )[
                "consumption_marker"
            ]
        )

        first_bytes = (
            marker.read_bytes()
        )

        require_raises(
            RuntimeError,
            lambda:
                preflight.claim_single_use_consumption(
                    execution_authority_commit=
                        (
                            "b"
                            *
                            40
                        ),

                    claimed_at_utc=
                        "2026-09-12T10:31:00Z",

                    artifact_dir=
                        root,
                ),
        )

        assert (
            marker.read_bytes()
            ==
            first_bytes
        )


def test_invalid_claim_authority_is_rejected_before_marker(
) -> None:

    with TemporaryDirectory() as directory:

        root = Path(
            directory
        )

        require_raises(
            ValueError,
            lambda:
                preflight.claim_single_use_consumption(
                    execution_authority_commit=
                        "not-a-git-sha",

                    claimed_at_utc=
                        "2026-09-12T10:30:00Z",

                    artifact_dir=
                        root,
                ),
        )

        assert (
            not
            preflight.single_use_artifact_paths(
                artifact_dir=
                    root
            )[
                "consumption_marker"
            ]
            .exists()
        )


def test_preflight_module_has_no_holdout_case_or_gold_authority(
) -> None:

    source = MODULE_PATH.read_text(
        encoding="utf-8"
    )

    assert (
        "HOLDOUT_CASES_PATH"
        not in
        source
    )

    assert (
        "holdout_v0.1_cases.json"
        not in
        source
    )

    assert (
        "gold_relation"
        not in
        source
    )

    assert (
        "gold_reason"
        not in
        source
    )


def test_preflight_module_has_no_model_execution_code(
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

    assert (
        "torch"
        not in
        imported_roots
    )

    assert (
        "transformers"
        not in
        imported_roots
    )

    assert (
        "peft"
        not in
        imported_roots
    )

    assert (
        "bitsandbytes"
        not in
        imported_roots
    )

    forbidden_call_attributes = {
        "from_pretrained",
        "generate",
    }

    forbidden_direct_call_names = {
        "PeftModel",
        "Gemma3ForCausalLM",
    }

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.Call,
        ):

            continue

        function = node.func

        if (
            isinstance(
                function,
                ast.Attribute,
            )
            and
            function.attr
            in
            forbidden_call_attributes
        ):

            raise AssertionError(
                (
                    "Preflight module must not call "
                    f"{function.attr}()."
                )
            )

        if (
            isinstance(
                function,
                ast.Name,
            )
            and
            function.id
            in
            forbidden_direct_call_names
        ):

            raise AssertionError(
                (
                    "Preflight module must not instantiate "
                    f"{function.id}."
                )
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
    test_frozen_authority_constants,
    test_committed_source_authority_is_platform_independent,
    test_source_eol_normalization,
    test_source_authority_rejects_non_eol_drift_contract,
    test_real_preflight_snapshot_without_consumption,
    test_single_use_availability_blocks_every_terminal_artifact,
    test_atomic_single_use_claim,
    test_second_claim_fails_closed,
    test_invalid_claim_authority_is_rejected_before_marker,
    test_preflight_module_has_no_holdout_case_or_gold_authority,
    test_preflight_module_has_no_model_execution_code,
    test_real_evaluation_directory_remains_unconsumed,
)


def main(
) -> None:

    print(
        "=== R8 HOSPITAL OFFICIAL PREFLIGHT / SINGLE-USE v0.1 ==="
    )

    print()

    for test in TESTS:

        test()

        print(
            f"[PASS] {test.__name__}"
        )

    print()

    print(
        "PASS - hospital official preflight / single-use v0.1"
    )


if __name__ == "__main__":

    main()
