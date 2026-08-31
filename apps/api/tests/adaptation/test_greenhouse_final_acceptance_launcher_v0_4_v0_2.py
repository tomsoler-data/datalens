from __future__ import annotations

import hashlib
import inspect
import tempfile
import unittest
from pathlib import Path

from app.adaptation import (
    greenhouse_final_acceptance_launcher_v0_4_v0_2 as launcher,
)


class GreenhouseFinalAcceptanceLauncherV02Tests(
    unittest.TestCase
):

    def test_protected_read_is_refused_before_marker(
        self,
    ) -> None:

        with tempfile.TemporaryDirectory() as directory:

            root = Path(
                directory
            )

            marker = (
                root
                / "consumption.json"
            )

            protected = (
                root
                / "protected.json"
            )

            payload = b'{"hello":"greenhouse"}\n'

            protected.write_bytes(
                payload
            )

            gate = (
                launcher.ProtectedMaterialGate(
                    marker_path=
                        marker,
                )
            )

            with self.assertRaises(
                RuntimeError
            ):

                gate.read_verified_bytes(
                    path=
                        protected,
                    expected_sha256=
                        hashlib.sha256(
                            payload
                        ).hexdigest(),
                    expected_size_bytes=
                        len(
                            payload
                        ),
                    label=
                        "synthetic protected material",
                )


    def test_consumption_marker_uses_exclusive_creation(
        self,
    ) -> None:

        with tempfile.TemporaryDirectory() as directory:

            marker = (
                Path(
                    directory
                )
                /
                "consumption.json"
            )

            launcher.create_consumption_marker(
                path=
                    marker,
                payload={
                    "status":
                        "consumption_started",
                },
            )

            self.assertTrue(
                marker.is_file()
            )

            with self.assertRaises(
                RuntimeError
            ):

                launcher.create_consumption_marker(
                    path=
                        marker,
                    payload={
                        "status":
                            "second_attempt",
                    },
                )


    def test_protected_read_allowed_after_marker(
        self,
    ) -> None:

        with tempfile.TemporaryDirectory() as directory:

            root = Path(
                directory
            )

            marker = (
                root
                /
                "consumption.json"
            )

            protected = (
                root
                /
                "protected.json"
            )

            payload = b'{"synthetic":true}\n'

            protected.write_bytes(
                payload
            )

            launcher.create_consumption_marker(
                path=
                    marker,
                payload={
                    "status":
                        "consumption_started",
                },
            )

            gate = (
                launcher.ProtectedMaterialGate(
                    marker_path=
                        marker,
                )
            )

            observed = (
                gate.read_verified_bytes(
                    path=
                        protected,
                    expected_sha256=
                        hashlib.sha256(
                            payload
                        ).hexdigest(),
                    expected_size_bytes=
                        len(
                            payload
                        ),
                    label=
                        "synthetic protected material",
                )
            )

            self.assertEqual(
                observed,
                payload,
            )


    def test_protected_sha_mismatch_fails_closed(
        self,
    ) -> None:

        with tempfile.TemporaryDirectory() as directory:

            root = Path(
                directory
            )

            marker = (
                root
                /
                "consumption.json"
            )

            protected = (
                root
                /
                "protected.bin"
            )

            payload = b"abc"

            protected.write_bytes(
                payload
            )

            launcher.create_consumption_marker(
                path=
                    marker,
                payload={
                    "status":
                        "consumption_started",
                },
            )

            gate = (
                launcher.ProtectedMaterialGate(
                    marker_path=
                        marker,
                )
            )

            with self.assertRaises(
                RuntimeError
            ):

                gate.read_verified_bytes(
                    path=
                        protected,
                    expected_sha256=
                        "0" * 64,
                    expected_size_bytes=
                        len(
                            payload
                        ),
                    label=
                        "synthetic protected material",
                )


    def test_protected_size_mismatch_fails_closed(
        self,
    ) -> None:

        with tempfile.TemporaryDirectory() as directory:

            root = Path(
                directory
            )

            marker = (
                root
                /
                "consumption.json"
            )

            protected = (
                root
                /
                "protected.bin"
            )

            payload = b"abc"

            protected.write_bytes(
                payload
            )

            launcher.create_consumption_marker(
                path=
                    marker,
                payload={
                    "status":
                        "consumption_started",
                },
            )

            gate = (
                launcher.ProtectedMaterialGate(
                    marker_path=
                        marker,
                )
            )

            with self.assertRaises(
                RuntimeError
            ):

                gate.read_verified_bytes(
                    path=
                        protected,
                    expected_sha256=
                        hashlib.sha256(
                            payload
                        ).hexdigest(),
                    expected_size_bytes=
                        len(
                            payload
                        )
                        +
                        1,
                    label=
                        "synthetic protected material",
                )


    def test_gold_projection_same_metric_state(
        self,
    ) -> None:

        relation = (
            launcher.project_gold_relation(
                same_concept=True,
                same_concept_family=True,
                same_domain=True,
                distinct_variants=True,
                compatible_units=True,
                derived_gap_compatible=True,
            )
        )

        self.assertEqual(
            relation,
            "same_metric_different_state",
        )


    def test_gold_projection_related_distinct(
        self,
    ) -> None:

        relation = (
            launcher.project_gold_relation(
                same_concept=False,
                same_concept_family=True,
                same_domain=True,
                distinct_variants=True,
                compatible_units=False,
                derived_gap_compatible=False,
            )
        )

        self.assertEqual(
            relation,
            "related_distinct_metric",
        )


    def test_gold_projection_unrelated(
        self,
    ) -> None:

        relation = (
            launcher.project_gold_relation(
                same_concept=False,
                same_concept_family=False,
                same_domain=False,
                distinct_variants=False,
                compatible_units=False,
                derived_gap_compatible=False,
            )
        )

        self.assertEqual(
            relation,
            "unrelated",
        )


    def test_gold_projection_uncertain(
        self,
    ) -> None:

        relation = (
            launcher.project_gold_relation(
                same_concept=False,
                same_concept_family=False,
                same_domain=True,
                distinct_variants=False,
                compatible_units=True,
                derived_gap_compatible=False,
            )
        )

        self.assertEqual(
            relation,
            "uncertain",
        )


    def test_gold_projection_invariant_fails_closed(
        self,
    ) -> None:

        with self.assertRaises(
            RuntimeError
        ):

            launcher.project_gold_relation(
                same_concept=True,
                same_concept_family=False,
                same_domain=True,
                distinct_variants=True,
                compatible_units=True,
                derived_gap_compatible=True,
            )


    def test_case_extractor_accepts_unique_nested_18_case_list(
        self,
    ) -> None:

        cases = [
            {
                "case_id":
                    f"synthetic:{index:03d}",

                "left_column":
                    f"left_{index}",

                "right_column":
                    f"right_{index}",
            }

            for index
            in range(
                1,
                19,
            )
        ]

        payload = {
            "metadata": {
                "safe":
                    True,
            },
            "container": {
                "pair_cases":
                    cases,
            },
        }

        extracted = (
            launcher.extract_protected_case_entries(
                payload
            )
        )

        self.assertEqual(
            len(
                extracted
            ),
            18,
        )

        self.assertEqual(
            extracted[
                0
            ][
                "case_id"
            ],
            "synthetic:001",
        )


    def test_case_extractor_rejects_wrong_count(
        self,
    ) -> None:

        cases = [
            {
                "case_id":
                    f"synthetic:{index:03d}",

                "left_column":
                    f"left_{index}",

                "right_column":
                    f"right_{index}",
            }

            for index
            in range(
                1,
                18,
            )
        ]

        with self.assertRaises(
            RuntimeError
        ):

            launcher.extract_protected_case_entries(
                {
                    "cases":
                        cases,
                }
            )


    def test_case_extractor_rejects_duplicate_case_ids(
        self,
    ) -> None:

        cases = [
            {
                "case_id":
                    (
                        "duplicate"
                        if index
                        in {
                            1,
                            2,
                        }
                        else
                        f"synthetic:{index:03d}"
                    ),

                "left_column":
                    f"left_{index}",

                "right_column":
                    f"right_{index}",
            }

            for index
            in range(
                1,
                19,
            )
        ]

        with self.assertRaises(
            RuntimeError
        ):

            launcher.extract_protected_case_entries(
                cases
            )


    def test_acceptance_gates_all_pass(
        self,
    ) -> None:

        adapted = {
            "accuracy":
                0.833333,

            "strict_json_validity_rate":
                1.0,

            "dangerous_false_positives":
                0,

            "safety_failures":
                0,
        }

        gates = (
            launcher.evaluate_acceptance_gates(
                adapted=
                    adapted,
                freeze_integrity_pass=
                    True,
                s3_regression_pass=
                    True,
            )
        )

        self.assertTrue(
            gates[
                "all_passed"
            ]
        )


    def test_acceptance_gate_accuracy_failure_is_terminal(
        self,
    ) -> None:

        adapted = {
            "accuracy":
                0.777778,

            "strict_json_validity_rate":
                1.0,

            "dangerous_false_positives":
                0,

            "safety_failures":
                0,
        }

        gates = (
            launcher.evaluate_acceptance_gates(
                adapted=
                    adapted,
                freeze_integrity_pass=
                    True,
                s3_regression_pass=
                    True,
            )
        )

        self.assertFalse(
            gates[
                "all_passed"
            ]
        )

        self.assertEqual(
            gates[
                "failure_action"
            ],
            "terminal_final_acceptance_failure_no_reexecution",
        )


    def test_official_execution_order_places_marker_before_protected_read(
        self,
    ) -> None:

        source = inspect.getsource(
            launcher.execute_greenhouse_final_acceptance
        )

        marker_index = source.index(
            "create_consumption_marker("
        )

        first_protected_read_index = source.index(
            ".read_verified_bytes("
        )

        self.assertLess(
            marker_index,
            first_protected_read_index,
        )


    def test_no_retry_loop_in_model_evaluation(
        self,
    ) -> None:

        source = inspect.getsource(
            launcher.evaluate_model_once
        )

        self.assertNotIn(
            "while ",
            source,
        )

        self.assertNotIn(
            "retry",
            source.lower(),
        )


    def test_official_execution_requires_future_v03_authorization(
        self,
    ) -> None:

        self.assertTrue(
            str(
                launcher.DEFAULT_EXECUTION_AUTHORIZATION_PATH
            ).endswith(
                "greenhouse_final_acceptance_execution_authorization_v0.3.json"
            )
        )


    def test_final_acceptance_output_revision_stays_v01(
        self,
    ) -> None:

        for value in (
            launcher.EXPECTED_CONSUMPTION_OUTPUT,
            launcher.EXPECTED_REPORT_OUTPUT,
            launcher.EXPECTED_RECEIPT_OUTPUT,
        ):

            self.assertIn(
                "_v0.1_",
                value,
            )

            self.assertNotIn(
                "_v0.2_",
                value,
            )

            self.assertNotIn(
                "_v0.3_",
                value,
            )


    def test_launcher_import_does_not_consume_greenhouse(
        self,
    ) -> None:

        source = inspect.getsource(
            launcher
        )

        self.assertIn(
            'if __name__ == "__main__"',
            source,
        )


    def test_launcher_v02_runner_v04_binding(
        self,
    ) -> None:

        self.assertEqual(
            launcher.GREENHOUSE_FINAL_ACCEPTANCE_LAUNCHER_RULE_VERSION,
            "qlora_v0.4_greenhouse_final_acceptance_launcher_v0.2",
        )

        self.assertTrue(
            str(
                launcher.RUNNER_PATH
            ).endswith(
                "greenhouse_final_acceptance_runner_v0_4_v0_4.py"
            )
        )

        self.assertEqual(
            launcher.EXPECTED_RUNNER_SHA256,
            (
                "8e922f46d65048ab6bcebeab8eca7e52"
                "b8bf83100524bea84658530039b40cea"
            ),
        )


    def test_prebuilt_prompt_records_preserve_order(
        self,
    ) -> None:

        cases = [
            {
                "case_id":
                    f"synthetic:{index:03d}",

                "left_dataset_id":
                    "synthetic:greenhouse",

                "right_dataset_id":
                    "synthetic:greenhouse",

                "left_column":
                    f"left_{index}",

                "right_column":
                    f"right_{index}",

                "expected_relation":
                    "uncertain",
            }

            for index
            in range(
                1,
                19,
            )
        ]


        profile_index = {}

        observed = []


        original = (
            launcher.runner
            .build_label_blind_user_message
        )


        def synthetic_builder(
            *,
            case_identity,
            profile_index,
        ):

            observed.append(
                dict(
                    case_identity
                )
            )

            return (
                "prebuilt:"
                +
                case_identity[
                    "left_column"
                ]
            )


        launcher.runner.build_label_blind_user_message = (
            synthetic_builder
        )


        try:

            records = (
                launcher.build_prebuilt_prompt_records(
                    cases=
                        cases,
                    profile_index=
                        profile_index,
                )
            )

        finally:

            launcher.runner.build_label_blind_user_message = (
                original
            )


        self.assertIsInstance(
            records,
            tuple,
        )

        self.assertEqual(
            len(
                records
            ),
            18,
        )

        self.assertEqual(
            len(
                observed
            ),
            18,
        )


        for index, record in enumerate(
            records,
            start=1,
        ):

            self.assertEqual(
                set(
                    record
                ),
                {
                    "case_identity",
                    "user_message",
                },
            )

            self.assertEqual(
                record[
                    "case_identity"
                ][
                    "left_column"
                ],
                f"left_{index}",
            )

            self.assertEqual(
                record[
                    "user_message"
                ],
                f"prebuilt:left_{index}",
            )


        launcher.validate_prebuilt_prompt_records(
            cases=
                cases,
            prebuilt_prompt_records=
                records,
        )


    def test_prebuilt_validation_rejects_identity_drift(
        self,
    ) -> None:

        cases = [
            {
                "case_id":
                    f"synthetic:{index:03d}",

                "left_dataset_id":
                    "synthetic:greenhouse",

                "right_dataset_id":
                    "synthetic:greenhouse",

                "left_column":
                    f"left_{index}",

                "right_column":
                    f"right_{index}",

                "expected_relation":
                    "uncertain",
            }

            for index
            in range(
                1,
                19,
            )
        ]


        records = [
            {
                "case_identity": {
                    "left_dataset_id":
                        "synthetic:greenhouse",

                    "right_dataset_id":
                        "synthetic:greenhouse",

                    "left_column":
                        f"left_{index}",

                    "right_column":
                        f"right_{index}",
                },

                "user_message":
                    f"message-{index}",
            }

            for index
            in range(
                1,
                19,
            )
        ]


        records[
            0
        ][
            "case_identity"
        ][
            "left_column"
        ] = "wrong-column"


        with self.assertRaises(
            RuntimeError
        ):

            launcher.validate_prebuilt_prompt_records(
                cases=
                    cases,
                prebuilt_prompt_records=
                    tuple(
                        records
                    ),
            )


    def test_evaluate_model_once_uses_only_prebuilt_messages(
        self,
    ) -> None:

        cases = [
            {
                "case_id":
                    f"synthetic:{index:03d}",

                "left_dataset_id":
                    "synthetic:greenhouse",

                "right_dataset_id":
                    "synthetic:greenhouse",

                "left_column":
                    f"left_{index}",

                "right_column":
                    f"right_{index}",

                "expected_relation":
                    "uncertain",
            }

            for index
            in range(
                1,
                19,
            )
        ]


        records = tuple(
            {
                "case_identity": {
                    "left_dataset_id":
                        "synthetic:greenhouse",

                    "right_dataset_id":
                        "synthetic:greenhouse",

                    "left_column":
                        f"left_{index}",

                    "right_column":
                        f"right_{index}",
                },

                "user_message":
                    f"prebuilt-message-{index}",
            }

            for index
            in range(
                1,
                19,
            )
        )


        observed = []


        original_new = (
            launcher.runner
            .generate_label_blind_case_from_user_message
        )

        original_old = (
            launcher.runner
            .generate_label_blind_case
        )


        def synthetic_generate(
            *,
            model,
            tokenizer,
            user_message,
            torch_module,
        ):

            observed.append(
                user_message
            )

            return {
                "strict_json_valid":
                    True,

                "relation":
                    "uncertain",

                "reason":
                    (
                        "Synthetic result uses only the supplied "
                        "prebuilt semantic message safely."
                    ),
            }


        def forbidden_old(
            **_kwargs,
        ):

            raise RuntimeError(
                "Legacy prompt rebuilding path called."
            )


        launcher.runner.generate_label_blind_case_from_user_message = (
            synthetic_generate
        )

        launcher.runner.generate_label_blind_case = (
            forbidden_old
        )


        try:

            result = (
                launcher.evaluate_model_once(
                    model=
                        object(),
                    tokenizer=
                        object(),
                    cases=
                        cases,
                    prebuilt_prompt_records=
                        records,
                    torch_module=
                        object(),
                )
            )

        finally:

            launcher.runner.generate_label_blind_case_from_user_message = (
                original_new
            )

            launcher.runner.generate_label_blind_case = (
                original_old
            )


        self.assertEqual(
            observed,
            [
                f"prebuilt-message-{index}"
                for index
                in range(
                    1,
                    19,
                )
            ],
        )

        self.assertEqual(
            result[
                "case_count"
            ],
            18,
        )

        self.assertEqual(
            result[
                "correct_count"
            ],
            18,
        )

        self.assertEqual(
            result[
                "accuracy"
            ],
            1.0,
        )


    def test_prompt_prebuild_occurs_before_model_import_and_load(
        self,
    ) -> None:

        source = inspect.getsource(
            launcher.execute_greenhouse_final_acceptance
        )


        build_index = source.index(
            "build_prebuilt_prompt_records("
        )

        validate_index = source.index(
            "validate_prebuilt_prompt_records("
        )

        torch_index = source.index(
            "import torch"
        )

        load_index = source.index(
            ".load_base_model("
        )


        self.assertLess(
            build_index,
            validate_index,
        )

        self.assertLess(
            validate_index,
            torch_index,
        )

        self.assertLess(
            torch_index,
            load_index,
        )


    def test_evaluation_has_no_prompt_rebuild_path(
        self,
    ) -> None:

        source = inspect.getsource(
            launcher.evaluate_model_once
        )


        self.assertIn(
            "generate_label_blind_case_from_user_message(",
            source,
        )

        self.assertNotIn(
            "runner.generate_label_blind_case(",
            source,
        )

        self.assertNotIn(
            "build_label_blind_user_message(",
            source,
        )

        self.assertNotIn(
            "profile_index",
            source,
        )


    def test_base_and_adapted_use_same_prebuilt_sequence(
        self,
    ) -> None:

        source = inspect.getsource(
            launcher.execute_greenhouse_final_acceptance
        )


        self.assertEqual(
            source.count(
                "evaluate_model_once("
            ),
            2,
        )

        # one validate call + base call + adapted call
        self.assertEqual(
            source.count(
                "prebuilt_prompt_records="
            ),
            3,
        )


if __name__ == "__main__":

    unittest.main(
        verbosity=2
    )
