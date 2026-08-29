from __future__ import annotations

import ast
import inspect

from pathlib import Path


print(
    "=== DATALENS QLORA v0.4 OFFICIAL DATASET FREEZE RUNNER TEST v0.1 ==="
)

print()


ROOT = Path.cwd().resolve()


from app.adaptation.training_dataset_freeze_v0_4 import (
    AIRPORT_CASES_RELATIVE_PATH,
    CANONICAL_DATASET_RELATIVE_PATH,
    CONTAMINATION_REPORT_RELATIVE_PATH,
    EXPECTED_BASE_PROTECTED_CORPUS_SHA256,
    EXPECTED_CANONICAL_SHA256,
    EXPECTED_CONTAMINATION_CANDIDATE_SHA256,
    EXPECTED_EXTENDED_PROTECTED_CORPUS_SHA256,
    EXPECTED_FORBIDDEN_PROVENANCE_INDEX_SHA256,
    EXPECTED_EXAMPLE_COUNT,
    FREEZE_ARTIFACT_RELATIVE_PATH,
    PROVENANCE_REPORT_RELATIVE_PATH,
    QLORA_V0_4_DATASET_FREEZE_RUNNER_RULE_VERSION,
    assert_official_outputs_absent,
    build_governance_preflight,
)


EXPECTED_RULE_VERSION = (
    "qlora_v0.4_training_dataset_freeze_runner_v0.1"
)


if (
    QLORA_V0_4_DATASET_FREEZE_RUNNER_RULE_VERSION
    !=
    EXPECTED_RULE_VERSION
):
    raise RuntimeError(
        "Official freeze runner rule version mismatch."
    )


print(
    "Runner rule version: PASS"
)


for relative_path in (
    CANONICAL_DATASET_RELATIVE_PATH,
    CONTAMINATION_REPORT_RELATIVE_PATH,
    PROVENANCE_REPORT_RELATIVE_PATH,
    FREEZE_ARTIFACT_RELATIVE_PATH,
):
    path = (
        ROOT
        /
        relative_path
    )

    if path.exists():
        raise RuntimeError(
            (
                "Official output exists before "
                "official freeze: "
                f"{path}"
            )
        )


assert_official_outputs_absent(
    repository_root=
        ROOT,
)


print(
    "Official outputs absent: PASS"
)


airport_path = (
    ROOT
    /
    AIRPORT_CASES_RELATIVE_PATH
)


if not airport_path.is_file():
    raise FileNotFoundError(
        airport_path
    )


print(
    "Airport protected evidence binding exists: PASS"
)


preflight_a = (
    build_governance_preflight(
        repository_root=
            ROOT,
    )
)


preflight_b = (
    build_governance_preflight(
        repository_root=
            ROOT,
    )
)


if (
    preflight_a[
        "canonical_sha256"
    ]
    !=
    EXPECTED_CANONICAL_SHA256
):
    raise RuntimeError(
        "Canonical SHA mismatch."
    )


if (
    preflight_a[
        "contamination_candidate_sha256"
    ]
    !=
    EXPECTED_CONTAMINATION_CANDIDATE_SHA256
):
    raise RuntimeError(
        "Contamination candidate SHA mismatch."
    )


if (
    preflight_a[
        "base_corpus_sha256"
    ]
    !=
    EXPECTED_BASE_PROTECTED_CORPUS_SHA256
):
    raise RuntimeError(
        "Base protected corpus SHA mismatch."
    )


if (
    preflight_a[
        "extended_corpus_sha256"
    ]
    !=
    EXPECTED_EXTENDED_PROTECTED_CORPUS_SHA256
):
    raise RuntimeError(
        "Extended protected corpus SHA mismatch."
    )


if (
    preflight_a[
        "forbidden_index_sha256"
    ]
    !=
    EXPECTED_FORBIDDEN_PROVENANCE_INDEX_SHA256
):
    raise RuntimeError(
        "Forbidden provenance index SHA mismatch."
    )


if (
    len(
        preflight_a[
            "examples"
        ]
    )
    !=
    EXPECTED_EXAMPLE_COUNT
):
    raise RuntimeError(
        "Canonical example count mismatch."
    )


if (
    preflight_a[
        "contamination_match_count"
    ]
    !=
    0
):
    raise RuntimeError(
        "Contamination preflight is not clean."
    )


if (
    preflight_a[
        "provenance_violation_count"
    ]
    !=
    0
):
    raise RuntimeError(
        "Provenance preflight is not clean."
    )


for key in (
    "canonical_sha256",
    "base_corpus_sha256",
    "extended_corpus_sha256",
    "contamination_candidate_sha256",
    "contamination_match_count",
    "provenance_violation_count",
    "forbidden_index_sha256",
    "protected_source_count_base",
    "protected_source_count_extended",
    "protected_text_fingerprint_count_extended",
    "protected_structured_fingerprint_count_extended",
):
    if (
        preflight_a[
            key
        ]
        !=
        preflight_b[
            key
        ]
    ):
        raise RuntimeError(
            (
                "Governance preflight is not "
                f"deterministic for {key}."
            )
        )


print()

print(
    "OFFICIAL PREFLIGHT"
)

print(
    f"  Canonical examples: {len(preflight_a['examples'])}"
)

print(
    (
        "  Canonical SHA256: "
        f"{preflight_a['canonical_sha256']}"
    )
)

print(
    (
        "  Contamination candidate SHA256: "
        f"{preflight_a['contamination_candidate_sha256']}"
    )
)

print(
    (
        "  Base corpus SHA256: "
        f"{preflight_a['base_corpus_sha256']}"
    )
)

print(
    (
        "  Extended corpus SHA256: "
        f"{preflight_a['extended_corpus_sha256']}"
    )
)

print(
    (
        "  Forbidden provenance index SHA256: "
        f"{preflight_a['forbidden_index_sha256']}"
    )
)

print(
    (
        "  Contamination matches: "
        f"{preflight_a['contamination_match_count']}"
    )
)

print(
    (
        "  Provenance violations: "
        f"{preflight_a['provenance_violation_count']}"
    )
)


# ------------------------------------------------------------
# STATIC SAFETY
# ------------------------------------------------------------


module = __import__(
    (
        "app.adaptation."
        "training_dataset_freeze_v0_4"
    ),
    fromlist=[
        "*",
    ],
)


module_source = inspect.getsource(
    module
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
        imported_module = (
            node.module
            or
            ""
        )

        if (
            imported_module.split(
                "."
            )[0]
            in
            heavy_modules
        ):
            raise RuntimeError(
                (
                    "Heavy ML import detected: "
                    f"{imported_module}"
                )
            )


if (
    "additional_protected_paths"
    not in
    module_source
):
    raise RuntimeError(
        "Runner does not bind additional protected paths."
    )


if (
    "AIRPORT_HOLDOUT_DATASET_ID"
    not in
    module_source
):
    raise RuntimeError(
        "Runner does not bind Airport provenance ID."
    )


print()

print(
    "SAFETY"
)

print(
    "  Heavy ML imports: False"
)

print(
    "  Additional protected-path binding: PASS"
)

print(
    "  Airport provenance binding: PASS"
)

print(
    "  Official canonical dataset written: False"
)

print(
    "  Official reports written: False"
)

print(
    "  Official freeze artifact written: False"
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
    "  Evaluation executed: False"
)

print(
    "  Training executed: False"
)

print(
    "  Final Acceptance evaluated: False"
)


print()

print(
    "DATALENS QLORA v0.4 OFFICIAL DATASET FREEZE RUNNER TEST v0.1: PASS"
)
