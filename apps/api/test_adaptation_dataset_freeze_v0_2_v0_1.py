from __future__ import annotations

import ast
import inspect
import json
import shutil
import tempfile

from pathlib import Path


print(
    "=== DATALENS ADAPTATION DATASET FREEZE v0.2 TEST v0.1 ==="
)

print()


ROOT = Path.cwd().resolve()


from app.adaptation import adaptation_dataset
from app.adaptation.adaptation_dataset_freeze_v0_2 import (
    ADAPTATION_DATASET_FREEZE_V0_2_RULE_VERSION,
    ADAPTATION_DATASET_INTEGRITY_V0_2_RULE_VERSION,
    AdaptationDatasetFreezeArtifactV0_2,
    freeze_adaptation_dataset_v0_2,
    load_adaptation_dataset_freeze_v0_2,
    verify_adaptation_dataset_freeze_v0_2,
    assert_adaptation_dataset_integrity_v0_2,
)


AIRPORT_RELATIVE = (
    "artifacts/adaptation/holdouts/"
    "datalens_semantic_qlora_v0.4_"
    "airport_ground_operations_holdout_v0.1_cases.json"
)


AIRPORT_PATH = (
    ROOT
    /
    AIRPORT_RELATIVE
)


if not AIRPORT_PATH.is_file():
    raise FileNotFoundError(
        AIRPORT_PATH
    )


signature = inspect.signature(
    freeze_adaptation_dataset_v0_2
)


if (
    "additional_protected_paths"
    not in
    signature.parameters
):
    raise RuntimeError(
        "v0.2 freezer does not expose "
        "additional_protected_paths."
    )


print(
    "additional_protected_paths parameter: PASS"
)


expected_fields = {
    "contamination_protected_corpus_sha256",
    "provenance_protected_corpus_sha256",
    "forbidden_provenance_index_sha256",
    "additional_protected_paths",
    "additional_forbidden_dataset_ids",
    "additional_forbidden_artifact_paths",
    "near_duplicate_threshold",
}


missing_fields = (
    expected_fields
    -
    set(
        AdaptationDatasetFreezeArtifactV0_2
        .model_fields
    )
)


if missing_fields:
    raise RuntimeError(
        (
            "Missing v0.2 freeze fields: "
            f"{sorted(missing_fields)}"
        )
    )


print(
    "Dual protected-corpus identity fields: PASS"
)

print(
    "Governance policy inputs persisted: PASS"
)


module_source = inspect.getsource(
    __import__(
        (
            "app.adaptation."
            "adaptation_dataset_freeze_v0_2"
        ),
        fromlist=[
            "*",
        ],
    )
)


tree = ast.parse(
    module_source
)


heavy_modules = {
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "safetensors",
}


for node in ast.walk(
    tree
):
    if isinstance(
        node,
        ast.Import,
    ):
        for alias in node.names:
            if (
                alias.name.split(
                    "."
                )[0]
                in
                heavy_modules
            ):
                raise RuntimeError(
                    (
                        "Heavy ML import detected: "
                        f"{alias.name}"
                    )
                )

    elif isinstance(
        node,
        ast.ImportFrom,
    ):
        module = (
            node.module
            or
            ""
        )

        if (
            module.split(
                "."
            )[0]
            in
            heavy_modules
        ):
            raise RuntimeError(
                (
                    "Heavy ML import detected: "
                    f"{module}"
                )
            )


print(
    "Heavy ML imports: False"
)


example = (
    adaptation_dataset.AdaptationTrainingExample(
        example_id=
            "freeze-v0.2:synthetic:001",

        messages=(
            adaptation_dataset.AdaptationMessage(
                role=
                    "user",

                content=(
                    "Freeze-v0.2 synthetic sentinel. "
                    "Compare a zaffre flux counter with "
                    "an umber latch counter in an invented "
                    "offline subsystem. Their definitions "
                    "describe separate synthetic quantities."
                ),
            ),

            adaptation_dataset.AdaptationMessage(
                role=
                    "assistant",

                content=json.dumps(
                    {
                        "relation":
                            "unrelated",

                        "reason": (
                            "Zaffre flux counts and umber "
                            "latch counts measure separate "
                            "synthetic quantities in this "
                            "isolated freezer verification sample."
                        ),
                    },
                    ensure_ascii=True,
                    separators=(
                        ",",
                        ":",
                    ),
                ),
            ),
        ),

        provenance=(
            adaptation_dataset.AdaptationExampleProvenance(
                origin=
                    "independently_authored",

                source_ids=(
                    "synthetic:freeze-v0.2:001",
                ),

                source_dataset_ids=
                    (),

                source_artifact_paths=
                    (),
            )
        ),

        tags=(
            "synthetic_freeze_test",
            "offline_only",
        ),
    )
)


TEMP_ROOT = (
    ROOT
    /
    "artifacts"
    /
    "adaptation"
    /
    "_freeze_v0_2_test_tmp"
)


TEMP_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


AIRPORT_DATASET_ID = (
    "adaptation:datalens-semantic-qlora-v0.4:"
    "airport-ground-operations:holdout:v0.1"
)


GREENHOUSE_DATASET_ID = (
    "greenhouse_operations:final_acceptance:0001"
)


try:
    with tempfile.TemporaryDirectory(
        prefix=
            "freeze_v0_2_",
        dir=
            str(
                TEMP_ROOT
            ),
    ) as temp_directory:
        temp_path = Path(
            temp_directory
        )

        dataset_path = (
            temp_path
            /
            "dataset.jsonl"
        )

        contamination_path = (
            temp_path
            /
            "contamination.json"
        )

        provenance_path = (
            temp_path
            /
            "provenance.json"
        )

        freeze_path = (
            temp_path
            /
            "freeze.json"
        )

        artifact = (
            freeze_adaptation_dataset_v0_2(
                repository_root=
                    ROOT,

                dataset_id=
                    "adaptation:test:freeze-v0.2:synthetic",

                dataset_version=
                    "adaptation_freeze_v0_2_synthetic_v0.1",

                examples=(
                    example,
                ),

                dataset_output_path=
                    dataset_path,

                contamination_report_output_path=
                    contamination_path,

                provenance_report_output_path=
                    provenance_path,

                freeze_output_path=
                    freeze_path,

                additional_protected_paths=(
                    AIRPORT_PATH,
                ),

                additional_forbidden_dataset_ids=(
                    AIRPORT_DATASET_ID,
                    GREENHOUSE_DATASET_ID,
                ),

                additional_forbidden_artifact_paths=(
                    AIRPORT_RELATIVE,
                ),

                near_duplicate_threshold=
                    0.92,

                training_has_started=
                    False,

                frozen_at=
                    "2026-08-30T00:00:00+00:00",
            )
        )

        if not all(
            path.is_file()

            for path
            in (
                dataset_path,
                contamination_path,
                provenance_path,
                freeze_path,
            )
        ):
            raise RuntimeError(
                "Atomic freeze bundle is incomplete."
            )

        if (
            artifact.freeze_rule_version
            !=
            ADAPTATION_DATASET_FREEZE_V0_2_RULE_VERSION
        ):
            raise RuntimeError(
                "Freeze rule version mismatch."
            )

        if (
            artifact.additional_protected_paths
            !=
            (
                AIRPORT_RELATIVE,
            )
        ):
            raise RuntimeError(
                "Airport protected-path binding mismatch."
            )

        if (
            artifact.contamination_match_count
            !=
            0
        ):
            raise RuntimeError(
                "Synthetic contamination gate failed."
            )

        if (
            artifact.provenance_violation_count
            !=
            0
        ):
            raise RuntimeError(
                "Synthetic provenance gate failed."
            )

        loaded = (
            load_adaptation_dataset_freeze_v0_2(
                freeze_path
            )
        )

        if loaded != artifact:
            raise RuntimeError(
                "Freeze artifact round-trip mismatch."
            )

        report = (
            verify_adaptation_dataset_freeze_v0_2(
                repository_root=
                    ROOT,

                freeze_path=
                    freeze_path,
            )
        )

        assert_adaptation_dataset_integrity_v0_2(
            report
        )

        if (
            report.rule_version
            !=
            ADAPTATION_DATASET_INTEGRITY_V0_2_RULE_VERSION
        ):
            raise RuntimeError(
                "Integrity rule version mismatch."
            )

        print()

        print(
            "FREEZE EXECUTION"
        )

        print(
            "  Synthetic canonical examples: 1"
        )

        print(
            "  Additional Airport content protection: PASS"
        )

        print(
            "  Airport forbidden provenance path: PASS"
        )

        print(
            "  Atomic four-file publication: PASS"
        )

        print(
            "  Freeze artifact round-trip: PASS"
        )

        print()

        print(
            "DUAL CORPUS IDENTITIES"
        )

        print(
            (
                "  Contamination corpus SHA256: "
                f"{artifact.contamination_protected_corpus_sha256}"
            )
        )

        print(
            (
                "  Provenance corpus SHA256: "
                f"{artifact.provenance_protected_corpus_sha256}"
            )
        )

        print(
            (
                "  Forbidden index SHA256: "
                f"{artifact.forbidden_provenance_index_sha256}"
            )
        )

        print()

        print(
            "INTEGRITY"
        )

        print(
            (
                "  Dataset hash: "
                f"{report.dataset_hash_valid}"
            )
        )

        print(
            (
                "  Contamination report hash: "
                f"{report.contamination_report_hash_valid}"
            )
        )

        print(
            (
                "  Provenance report hash: "
                f"{report.provenance_report_hash_valid}"
            )
        )

        print(
            (
                "  Contamination gate: "
                f"{report.contamination_gate_valid}"
            )
        )

        print(
            (
                "  Provenance gate: "
                f"{report.provenance_gate_valid}"
            )
        )

        print(
            (
                "  Contamination corpus binding: "
                f"{report.contamination_corpus_binding_valid}"
            )
        )

        print(
            (
                "  Provenance index binding: "
                f"{report.provenance_index_binding_valid}"
            )
        )

        print(
            (
                "  Evidence consistent: "
                f"{report.evidence_consistent}"
            )
        )

        print(
            (
                "  Passed: "
                f"{report.passed}"
            )
        )

finally:
    if TEMP_ROOT.exists():
        try:
            TEMP_ROOT.rmdir()
        except OSError:
            shutil.rmtree(
                TEMP_ROOT,
                ignore_errors=True,
            )


print()

print(
    "SAFETY"
)

print(
    "  Official v0.4 dataset frozen: False"
)

print(
    "  Airport evaluated: False"
)

print(
    "  Airport results observed: False"
)

print(
    "  Model loaded: False"
)

print(
    "  Adapter loaded: False"
)

print(
    "  CUDA requested: False"
)

print(
    "  Training executed: False"
)

print(
    "  Final Acceptance evaluated: False"
)


print()

print(
    "DATALENS ADAPTATION DATASET FREEZE v0.2 TEST v0.1: PASS"
)
