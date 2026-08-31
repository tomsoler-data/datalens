from __future__ import annotations


import inspect
import unittest

from collections import Counter
from pathlib import Path
from typing import Any


from app.adaptation import (
    greenhouse_final_acceptance_launcher_v0_4_v0_3
    as launcher,
)

from app.evaluation.benchmarks import (
    greenhouse_operations_final_acceptance
    as frozen_benchmark,
)


GOLD_KEYS = {
    "same_concept",
    "same_concept_family",
    "same_domain",
    "distinct_variants",
    "compatible_units",
    "derived_gap_compatible",
}


IDENTITY_KEYS = {
    "case_id",
    "left_dataset_id",
    "right_dataset_id",
    "left_column",
    "right_column",
}


def metadata_only_fixture(
) -> dict[
    str,
    Any
]:

    return {
        "adaptation_tuning_input":
            False,

        "benchmark_id":
            frozen_benchmark
            .GREENHOUSE_FINAL_ACCEPTANCE_BENCHMARK_ID,

        "benchmark_version":
            frozen_benchmark
            .GREENHOUSE_FINAL_ACCEPTANCE_BENCHMARK_VERSION,

        "column_case_count":
            0,

        "dataset_column_count":
            len(
                frozen_benchmark.COLUMNS
            ),

        "dataset_filename":
            frozen_benchmark
            .GREENHOUSE_FINAL_ACCEPTANCE_FILENAME,

        "dataset_id":
            frozen_benchmark
            .GREENHOUSE_FINAL_ACCEPTANCE_DATASET_ID,

        "dataset_row_count":
            len(
                frozen_benchmark.ROWS
            ),

        "domain":
            "commercial_greenhouse_operations",

        "evaluation_role":
            "independent_final_acceptance_holdout",

        "frozen_before_training":
            True,

        "independence_scan": {
            "candidate_material_count":
                0,

            "match_count":
                0,

            "protected_source_count":
                0,

            "protected_structured_fingerprint_count":
                0,

            "protected_text_fingerprint_count":
                0,

            "scan_status":
                "synthetic_regression_fixture",

            "candidate_overlap_detected":
                False,

            "training_overlap_detected":
                False,

            "adapter_overlap_detected":
                False,

            "manual_override":
                False,

            "independent":
                True,
        },

        "independent_final_evidence":
            True,

        "methodology": {
            "case_source":
                "frozen_benchmark_PAIR_CASES",

            "manifest_role":
                "metadata_only",

            "synthetic_fixture":
                True,
        },

        "pair_case_count":
            18,

        "preregistered_gates": {
            "accuracy_minimum":
                0.8,

            "strict_json_validity_required":
                1.0,

            "dangerous_false_positives_maximum":
                0,

            "safety_failures_maximum":
                0,

            "freeze_integrity_required":
                True,
        },

        "technical_split":
            "holdout",

        "total_case_count":
            18,

        "training_started":
            False,
    }


class GreenhouseFinalAcceptanceLauncherV03Tests(
    unittest.TestCase
):

    def test_metadata_only_manifest_is_accepted(
        self,
    ) -> None:

        payload = (
            metadata_only_fixture()
        )


        validated = (
            launcher
            .validate_metadata_only_case_manifest(
                payload,
                benchmark_module=
                    frozen_benchmark,
            )
        )


        self.assertEqual(
            validated,
            payload,
        )


    def test_metadata_only_manifest_rejects_any_json_list(
        self,
    ) -> None:

        payload = (
            metadata_only_fixture()
        )


        payload[
            "unexpected_list"
        ] = []


        with self.assertRaises(
            RuntimeError
        ):

            launcher.validate_metadata_only_case_manifest(
                payload,
                benchmark_module=
                    frozen_benchmark,
            )


    def test_metadata_only_manifest_rejects_case_level_fields(
        self,
    ) -> None:

        payload = (
            metadata_only_fixture()
        )


        payload[
            "case_id"
        ] = "forbidden"


        with self.assertRaises(
            RuntimeError
        ):

            launcher.validate_metadata_only_case_manifest(
                payload,
                benchmark_module=
                    frozen_benchmark,
            )


    def test_metadata_only_manifest_rejects_benchmark_identity_drift(
        self,
    ) -> None:

        payload = (
            metadata_only_fixture()
        )


        payload[
            "benchmark_id"
        ] = "semantic:wrong"


        with self.assertRaisesRegex(
            RuntimeError,
            "benchmark_id",
        ):

            launcher.validate_metadata_only_case_manifest(
                payload,
                benchmark_module=
                    frozen_benchmark,
            )


    def test_metadata_only_manifest_rejects_dataset_identity_drift(
        self,
    ) -> None:

        payload = (
            metadata_only_fixture()
        )


        payload[
            "dataset_id"
        ] = "wrong:dataset"


        with self.assertRaisesRegex(
            RuntimeError,
            "dataset_id",
        ):

            launcher.validate_metadata_only_case_manifest(
                payload,
                benchmark_module=
                    frozen_benchmark,
            )


    def test_metadata_only_manifest_rejects_case_count_drift(
        self,
    ) -> None:

        payload = (
            metadata_only_fixture()
        )


        payload[
            "total_case_count"
        ] = 17


        with self.assertRaisesRegex(
            RuntimeError,
            "total_case_count",
        ):

            launcher.validate_metadata_only_case_manifest(
                payload,
                benchmark_module=
                    frozen_benchmark,
            )


    def test_metadata_only_manifest_rejects_pair_count_drift(
        self,
    ) -> None:

        payload = (
            metadata_only_fixture()
        )


        payload[
            "pair_case_count"
        ] = 17


        with self.assertRaisesRegex(
            RuntimeError,
            "pair_case_count",
        ):

            launcher.validate_metadata_only_case_manifest(
                payload,
                benchmark_module=
                    frozen_benchmark,
            )


    def test_metadata_only_manifest_rejects_column_count_drift(
        self,
    ) -> None:

        payload = (
            metadata_only_fixture()
        )


        payload[
            "dataset_column_count"
        ] = 17


        with self.assertRaisesRegex(
            RuntimeError,
            "dataset_column_count",
        ):

            launcher.validate_metadata_only_case_manifest(
                payload,
                benchmark_module=
                    frozen_benchmark,
            )


    def test_metadata_only_manifest_rejects_row_count_drift(
        self,
    ) -> None:

        payload = (
            metadata_only_fixture()
        )


        payload[
            "dataset_row_count"
        ] = 11


        with self.assertRaisesRegex(
            RuntimeError,
            "dataset_row_count",
        ):

            launcher.validate_metadata_only_case_manifest(
                payload,
                benchmark_module=
                    frozen_benchmark,
            )


    def test_metadata_only_manifest_rejects_tuning_flag_drift(
        self,
    ) -> None:

        payload = (
            metadata_only_fixture()
        )


        payload[
            "adaptation_tuning_input"
        ] = True


        with self.assertRaises(
            RuntimeError
        ):

            launcher.validate_metadata_only_case_manifest(
                payload,
                benchmark_module=
                    frozen_benchmark,
            )


    def test_metadata_only_manifest_rejects_independence_match(
        self,
    ) -> None:

        payload = (
            metadata_only_fixture()
        )


        payload[
            "independence_scan"
        ][
            "match_count"
        ] = 1


        with self.assertRaises(
            RuntimeError
        ):

            launcher.validate_metadata_only_case_manifest(
                payload,
                benchmark_module=
                    frozen_benchmark,
            )


    def test_benchmark_identity_builder_returns_exactly_18_cases(
        self,
    ) -> None:

        identities = (
            launcher
            .build_benchmark_case_identity_entries(
                benchmark_module=
                    frozen_benchmark,
            )
        )


        self.assertEqual(
            len(
                identities
            ),
            18,
        )


        self.assertEqual(
            len(
                {
                    item[
                        "case_id"
                    ]

                    for item
                    in identities
                }
            ),
            18,
        )


    def test_benchmark_identity_builder_is_label_blind(
        self,
    ) -> None:

        identities = (
            launcher
            .build_benchmark_case_identity_entries(
                benchmark_module=
                    frozen_benchmark,
            )
        )


        for identity in identities:

            self.assertEqual(
                set(
                    identity.keys()
                ),
                IDENTITY_KEYS,
            )


            self.assertTrue(
                GOLD_KEYS.isdisjoint(
                    identity.keys()
                )
            )


    def test_gold_builder_accepts_benchmark_owned_identity_entries(
        self,
    ) -> None:

        identities = (
            launcher
            .build_benchmark_case_identity_entries(
                benchmark_module=
                    frozen_benchmark,
            )
        )


        gold_cases = (
            launcher
            .build_gold_case_records(
                benchmark_module=
                    frozen_benchmark,
                protected_case_entries=
                    identities,
            )
        )


        self.assertEqual(
            len(
                gold_cases
            ),
            18,
        )


        distribution = Counter(
            case[
                "expected_relation"
            ]

            for case
            in gold_cases
        )


        self.assertEqual(
            distribution[
                "same_metric_different_state"
            ],
            6,
        )


        self.assertEqual(
            distribution[
                "related_distinct_metric"
            ],
            3,
        )


        self.assertEqual(
            distribution[
                "uncertain"
            ],
            9,
        )


        self.assertEqual(
            distribution[
                "unrelated"
            ],
            0,
        )


    def test_historical_extractor_is_retained_but_not_used_by_execute(
        self,
    ) -> None:

        self.assertTrue(
            callable(
                launcher
                .extract_protected_case_entries
            )
        )


        execute_source = inspect.getsource(
            launcher
            .execute_greenhouse_final_acceptance
        )


        self.assertNotIn(
            "extract_protected_case_entries(",
            execute_source,
        )


        self.assertIn(
            "validate_metadata_only_case_manifest(",
            execute_source,
        )


        self.assertIn(
            "build_benchmark_case_identity_entries(",
            execute_source,
        )


    def test_single_use_barrier_is_preserved(
        self,
    ) -> None:

        execute_source = inspect.getsource(
            launcher
            .execute_greenhouse_final_acceptance
        )


        self.assertIn(
            "require_single_use_outputs_absent(",
            execute_source,
        )


        self.assertIn(
            "create_consumption_marker(",
            execute_source,
        )


    def test_launcher_path_uses_v03_filename_when_explicit(
        self,
    ) -> None:

        path = Path(
            launcher.LAUNCHER_PATH
        )


        self.assertEqual(
            path.name,
            (
                "greenhouse_final_acceptance_"
                "launcher_v0_4_v0_3.py"
            ),
        )


if __name__ == "__main__":

    unittest.main(
        verbosity=2
    )
