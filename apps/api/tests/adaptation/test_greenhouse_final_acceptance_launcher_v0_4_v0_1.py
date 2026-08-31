from __future__ import annotations

import hashlib
import inspect
import tempfile
import unittest
from pathlib import Path

from app.adaptation import (
    greenhouse_final_acceptance_launcher_v0_4_v0_1 as launcher,
)


class GreenhouseFinalAcceptanceLauncherV01Tests(
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


    def test_official_execution_requires_future_v02_authorization(
        self,
    ) -> None:

        self.assertTrue(
            str(
                launcher.DEFAULT_EXECUTION_AUTHORIZATION_PATH
            ).endswith(
                "greenhouse_final_acceptance_execution_authorization_v0.2.json"
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


if __name__ == "__main__":

    unittest.main(
        verbosity=2
    )
