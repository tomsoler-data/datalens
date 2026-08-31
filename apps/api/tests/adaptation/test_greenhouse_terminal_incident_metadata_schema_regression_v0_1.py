from __future__ import annotations


import unittest

from typing import (
    Any,
)


from app.adaptation import (
    greenhouse_final_acceptance_launcher_v0_4_v0_2
    as historical_launcher,
)

from app.evaluation.benchmarks import (
    greenhouse_operations_final_acceptance
    as frozen_benchmark,
)


# ============================================================
# DATALENS QLORA v0.4
#
# Greenhouse Terminal Incident
# Metadata-only Case Manifest Regression Reproducer v0.1
#
# IMPORTANT:
#
# This suite intentionally targets the HISTORICAL frozen
# Launcher v0.2 involved in the consumed Greenhouse v0.1
# terminal incident.
#
# It does NOT represent the desired future behavior.
#
# It proves:
#
# 1. the real case-manifest contract can be metadata-only;
# 2. such a manifest contains no JSON case list;
# 3. historical Launcher v0.2 rejects it with zero candidates;
# 4. the old synthetic nested-list fixture succeeds;
# 5. the canonical 18 pair cases already exist in the frozen
#    benchmark PAIR_CASES authority;
# 6. metadata and benchmark can be identity/count bound
#    without duplicating the case list into the manifest.
#
# No protected Greenhouse input is read by this test.
# No model/runtime execution occurs.
# ============================================================


EXPECTED_METADATA_KEYS = {
    "adaptation_tuning_input",
    "benchmark_id",
    "benchmark_version",
    "column_case_count",
    "dataset_column_count",
    "dataset_filename",
    "dataset_id",
    "dataset_row_count",
    "domain",
    "evaluation_role",
    "frozen_before_training",
    "independence_scan",
    "independent_final_evidence",
    "methodology",
    "pair_case_count",
    "preregistered_gates",
    "technical_split",
    "total_case_count",
    "training_started",
}


GOLD_ASSERTION_KEYS = {
    "same_concept",
    "same_concept_family",
    "same_domain",
    "distinct_variants",
    "compatible_units",
    "derived_gap_compatible",
}


def contains_json_list(
    value: Any,
) -> bool:

    if isinstance(
        value,
        list,
    ):

        return True


    if isinstance(
        value,
        dict,
    ):

        return any(
            contains_json_list(
                child
            )

            for child
            in value.values()
        )


    return False


def collect_keys(
    value: Any,
) -> set[
    str
]:

    result: set[
        str
    ] = set()


    if isinstance(
        value,
        dict,
    ):

        for key, child in value.items():

            result.add(
                str(
                    key
                )
            )

            result.update(
                collect_keys(
                    child
                )
            )


    elif isinstance(
        value,
        list,
    ):

        for child in value:

            result.update(
                collect_keys(
                    child
                )
            )


    return result


def metadata_only_case_manifest_fixture(
) -> dict[
    str,
    Any
]:
    """
    Synthetic metadata-only replica of the consumed
    Greenhouse case-manifest TOP-LEVEL contract.

    It deliberately contains zero JSON lists and no
    model-evaluation gold assertions.

    Values are synthetic/non-protected.
    """

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
            18,

        "dataset_filename":
            frozen_benchmark
            .GREENHOUSE_FINAL_ACCEPTANCE_FILENAME,

        "dataset_id":
            frozen_benchmark
            .GREENHOUSE_FINAL_ACCEPTANCE_DATASET_ID,

        "dataset_row_count":
            12,

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

            "case_materialization":
                "benchmark_owned",

            "manifest_role":
                "metadata_only",

            "pair_identity_source":
                "frozen_benchmark",

            "gold_source":
                "frozen_benchmark",

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


def legacy_nested_case_list_fixture(
) -> dict[
    str,
    Any
]:

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


    return {
        "metadata": {
            "safe":
                True,
        },

        "container": {
            "pair_cases":
                cases,
        },
    }


class GreenhouseMetadataOnlyIncidentRegressionV01Tests(
    unittest.TestCase
):

    def test_metadata_fixture_reproduces_top_level_manifest_contract(
        self,
    ) -> None:

        payload = (
            metadata_only_case_manifest_fixture()
        )


        self.assertEqual(
            set(
                payload.keys()
            ),
            EXPECTED_METADATA_KEYS,
        )


        self.assertEqual(
            payload[
                "total_case_count"
            ],
            18,
        )


        self.assertEqual(
            payload[
                "pair_case_count"
            ],
            18,
        )


        self.assertEqual(
            payload[
                "column_case_count"
            ],
            0,
        )


        self.assertEqual(
            payload[
                "dataset_column_count"
            ],
            18,
        )


        self.assertEqual(
            payload[
                "dataset_row_count"
            ],
            12,
        )


    def test_metadata_fixture_contains_zero_json_lists(
        self,
    ) -> None:

        payload = (
            metadata_only_case_manifest_fixture()
        )


        self.assertFalse(
            contains_json_list(
                payload
            )
        )


    def test_metadata_fixture_contains_no_case_level_gold_assertions(
        self,
    ) -> None:

        payload = (
            metadata_only_case_manifest_fixture()
        )


        observed_keys = collect_keys(
            payload
        )


        self.assertTrue(
            GOLD_ASSERTION_KEYS.isdisjoint(
                observed_keys
            )
        )


    def test_historical_v02_reproduces_terminal_zero_candidate_failure(
        self,
    ) -> None:

        payload = (
            metadata_only_case_manifest_fixture()
        )


        with self.assertRaisesRegex(
            RuntimeError,
            r"found 0\.",
        ):

            historical_launcher.extract_protected_case_entries(
                payload
            )


    def test_historical_v02_accepts_old_nested_list_fixture(
        self,
    ) -> None:

        payload = (
            legacy_nested_case_list_fixture()
        )


        extracted = (
            historical_launcher.extract_protected_case_entries(
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


        self.assertEqual(
            extracted[
                17
            ][
                "case_id"
            ],
            "synthetic:018",
        )


    def test_frozen_benchmark_is_canonical_source_of_18_pair_cases(
        self,
    ) -> None:

        cases = (
            frozen_benchmark
            .build_greenhouse_final_acceptance_pair_cases()
        )


        self.assertEqual(
            len(
                cases
            ),
            18,
        )


        case_ids = [
            str(
                case.case_id
            )

            for case
            in cases
        ]


        self.assertEqual(
            len(
                set(
                    case_ids
                )
            ),
            18,
        )


        self.assertEqual(
            len(
                frozen_benchmark.COLUMNS
            ),
            18,
        )


        self.assertEqual(
            len(
                frozen_benchmark.ROWS
            ),
            12,
        )


    def test_metadata_manifest_identity_matches_frozen_benchmark(
        self,
    ) -> None:

        payload = (
            metadata_only_case_manifest_fixture()
        )


        cases = (
            frozen_benchmark
            .build_greenhouse_final_acceptance_pair_cases()
        )


        self.assertEqual(
            payload[
                "benchmark_id"
            ],
            frozen_benchmark
            .GREENHOUSE_FINAL_ACCEPTANCE_BENCHMARK_ID,
        )


        self.assertEqual(
            payload[
                "benchmark_version"
            ],
            frozen_benchmark
            .GREENHOUSE_FINAL_ACCEPTANCE_BENCHMARK_VERSION,
        )


        self.assertEqual(
            payload[
                "dataset_id"
            ],
            frozen_benchmark
            .GREENHOUSE_FINAL_ACCEPTANCE_DATASET_ID,
        )


        self.assertEqual(
            payload[
                "dataset_filename"
            ],
            frozen_benchmark
            .GREENHOUSE_FINAL_ACCEPTANCE_FILENAME,
        )


        self.assertEqual(
            payload[
                "total_case_count"
            ],
            len(
                cases
            ),
        )


        self.assertEqual(
            payload[
                "pair_case_count"
            ],
            len(
                cases
            ),
        )


    def test_regression_fixture_demonstrates_exact_control_gap(
        self,
    ) -> None:

        metadata_payload = (
            metadata_only_case_manifest_fixture()
        )


        old_synthetic_payload = (
            legacy_nested_case_list_fixture()
        )


        self.assertFalse(
            contains_json_list(
                metadata_payload
            )
        )


        self.assertTrue(
            contains_json_list(
                old_synthetic_payload
            )
        )


        with self.assertRaises(
            RuntimeError
        ):

            historical_launcher.extract_protected_case_entries(
                metadata_payload
            )


        extracted = (
            historical_launcher.extract_protected_case_entries(
                old_synthetic_payload
            )
        )


        self.assertEqual(
            len(
                extracted
            ),
            18,
        )


if __name__ == "__main__":

    unittest.main(
        verbosity=2
    )
