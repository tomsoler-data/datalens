from __future__ import annotations


import hashlib
import json
import re


from collections.abc import Mapping


from pathlib import (
    Path,
    PurePosixPath,
)


from typing import Any


from app.model_lifecycle.artifact_contracts import (
    ModelLifecycleArtifactRecord,
)

from app.model_lifecycle.fingerprints import (
    source_contract_sha256,
)


# ============================================================
# VERSION
# ============================================================


QLORA_LIFECYCLE_PROJECTION_RULE_VERSION = (
    "qlora_lifecycle_projection_v0.1"
)


# ============================================================
# ERROR
# ============================================================


class QLoRALifecycleProjectionError(
    RuntimeError
):
    pass


# ============================================================
# VALIDATION HELPERS
# ============================================================


SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)


def _required_text(
    value: object,
    *,
    field_name: str,
) -> str:

    normalized = str(
        value
        if value is not None
        else ""
    ).strip()


    if not normalized:
        raise QLoRALifecycleProjectionError(
            f"{field_name} cannot be empty."
        )


    return normalized


def _require_mapping(
    value: object,
    *,
    field_name: str,
) -> Mapping[str, Any]:

    if not isinstance(
        value,
        Mapping,
    ):
        raise QLoRALifecycleProjectionError(
            f"{field_name} must be an object."
        )


    return value


def _require_sha256(
    value: object,
    *,
    field_name: str,
) -> str:

    normalized = (
        _required_text(
            value,
            field_name=field_name,
        )
        .lower()
    )


    if (
        SHA256_PATTERN.fullmatch(
            normalized
        )
        is None
    ):
        raise QLoRALifecycleProjectionError(
            (
                f"{field_name} must be "
                "a 64-character SHA-256 digest."
            )
        )


    return normalized


def _require_positive_int(
    value: object,
    *,
    field_name: str,
) -> int:

    if (
        isinstance(
            value,
            bool,
        )
        or
        not isinstance(
            value,
            int,
        )
        or
        value <= 0
    ):
        raise QLoRALifecycleProjectionError(
            (
                f"{field_name} must be "
                "a positive integer."
            )
        )


    return value


def _normalize_relative_path(
    value: object,
    *,
    field_name: str,
) -> str:

    normalized = (
        _required_text(
            value,
            field_name=field_name,
        )
        .replace(
            "\\",
            "/",
        )
    )


    path = PurePosixPath(
        normalized
    )


    if (
        path.is_absolute()
        or
        ".." in path.parts
    ):
        raise QLoRALifecycleProjectionError(
            (
                f"{field_name} must remain "
                "relative and non-escaping."
            )
        )


    return path.as_posix()


# ============================================================
# JSON / BYTE AUTHORITIES
# ============================================================


def _load_json_object(
    path: Path,
) -> dict[str, Any]:

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8-sig"
            )
        )

    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        raise QLoRALifecycleProjectionError(
            f"Could not load JSON authority: {path}"
        ) from error


    if not isinstance(
        payload,
        dict,
    ):
        raise QLoRALifecycleProjectionError(
            (
                "JSON authority must be "
                f"an object: {path}"
            )
        )


    return payload


def _lf_normalized_sha256(
    path: Path,
) -> str:

    try:
        raw = path.read_bytes()

    except OSError as error:
        raise QLoRALifecycleProjectionError(
            f"Could not read authority: {path}"
        ) from error


    if raw.startswith(
        b"\xef\xbb\xbf"
    ):
        raw = raw[
            3:
        ]


    try:
        text = raw.decode(
            "utf-8"
        )

    except UnicodeDecodeError as error:
        raise QLoRALifecycleProjectionError(
            (
                "Authority is not strict UTF-8: "
                f"{path}"
            )
        ) from error


    normalized = (
        text
        .replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
        .encode(
            "utf-8"
        )
    )


    return hashlib.sha256(
        normalized
    ).hexdigest()


# ============================================================
# PROJECTION
# ============================================================


def project_qlora_adapter_artifact(
    *,
    experiment_contract_path: Path,
    experiment_contract_freeze_path: Path,
    training_manifest_path: Path,
    training_receipt_path: Path,
) -> ModelLifecycleArtifactRecord:
    """
    Project already-frozen QLoRA training authorities into the
    neutral DataLens model-lifecycle artifact identity.

    This function does not:
    - train or tune a model;
    - load base-model or adapter weights;
    - read protected holdout cases;
    - copy or mutate the adapter artifact.

    Historical byte-level contract/manifest authorities are
    verified first. The shared source_contract_sha256 field then
    uses the lifecycle's formatting-independent canonical
    contract fingerprint.
    """

    contract = _load_json_object(
        experiment_contract_path
    )

    freeze = _load_json_object(
        experiment_contract_freeze_path
    )

    manifest = _load_json_object(
        training_manifest_path
    )

    receipt = _load_json_object(
        training_receipt_path
    )


    # ========================================================
    # TRAINING RECEIPT AUTHORITY
    # ========================================================


    if receipt.get(
        "passed"
    ) is not True:
        raise QLoRALifecycleProjectionError(
            "QLoRA training receipt did not pass."
        )


    if (
        receipt.get(
            "rule_version"
        )
        !=
        "qlora_training_receipt_v0.4_v0.1"
    ):
        raise QLoRALifecycleProjectionError(
            "Unexpected QLoRA training receipt rule."
        )


    experiment_id = _required_text(
        receipt.get(
            "experiment_id"
        ),
        field_name="receipt.experiment_id",
    )


    # ========================================================
    # EXPERIMENT CONTRACT AUTHORITY
    # ========================================================


    if (
        contract.get(
            "adaptation_method"
        )
        !=
        "qlora"
    ):
        raise QLoRALifecycleProjectionError(
            "Expected QLoRA adaptation method."
        )


    contract_rule = _required_text(
        contract.get(
            "rule_version"
        ),
        field_name="contract.rule_version",
    )


    if (
        contract_rule
        !=
        "qlora_experiment_contract_v0.1"
    ):
        raise QLoRALifecycleProjectionError(
            "Unexpected QLoRA experiment contract rule."
        )


    if (
        contract.get(
            "experiment_id"
        )
        !=
        experiment_id
    ):
        raise QLoRALifecycleProjectionError(
            (
                "Experiment contract and training "
                "receipt identity mismatch."
            )
        )


    # ========================================================
    # CONTRACT FREEZE AUTHORITY
    # ========================================================


    if (
        freeze.get(
            "status"
        )
        !=
        "frozen"
    ):
        raise QLoRALifecycleProjectionError(
            "Experiment contract is not frozen."
        )


    if (
        freeze.get(
            "experiment_id"
        )
        !=
        experiment_id
    ):
        raise QLoRALifecycleProjectionError(
            (
                "Experiment freeze identity "
                "does not match receipt."
            )
        )


    if (
        freeze.get(
            "experiment_contract_rule_version"
        )
        !=
        contract_rule
    ):
        raise QLoRALifecycleProjectionError(
            (
                "Experiment freeze contract rule "
                "does not match source contract."
            )
        )


    frozen_contract_sha = (
        _require_sha256(
            freeze.get(
                "contract_sha256"
            ),
            field_name=
                "freeze.contract_sha256",
        )
    )


    actual_contract_sha = (
        _lf_normalized_sha256(
            experiment_contract_path
        )
    )


    if (
        actual_contract_sha
        !=
        frozen_contract_sha
    ):
        raise QLoRALifecycleProjectionError(
            (
                "Experiment contract bytes do "
                "not match frozen authority."
            )
        )


    # ========================================================
    # TRAINING MANIFEST AUTHORITY
    # ========================================================


    if (
        manifest.get(
            "rule_version"
        )
        !=
        "training_execution_manifest_v0.2"
    ):
        raise QLoRALifecycleProjectionError(
            "Unexpected training manifest rule."
        )


    manifest_experiment = (
        _require_mapping(
            manifest.get(
                "experiment"
            ),
            field_name="manifest.experiment",
        )
    )


    if (
        manifest_experiment.get(
            "experiment_id"
        )
        !=
        experiment_id
    ):
        raise QLoRALifecycleProjectionError(
            (
                "Training manifest experiment "
                "identity mismatch."
            )
        )


    manifest_contract_sha = (
        _require_sha256(
            manifest_experiment.get(
                "experiment_contract_sha256"
            ),
            field_name=(
                "manifest.experiment."
                "experiment_contract_sha256"
            ),
        )
    )


    if (
        manifest_contract_sha
        !=
        frozen_contract_sha
    ):
        raise QLoRALifecycleProjectionError(
            (
                "Training manifest does not bind "
                "the frozen experiment contract."
            )
        )


    receipt_manifest_sha = (
        _require_sha256(
            receipt.get(
                "manifest_sha256"
            ),
            field_name=
                "receipt.manifest_sha256",
        )
    )


    actual_manifest_sha = (
        _lf_normalized_sha256(
            training_manifest_path
        )
    )


    if (
        actual_manifest_sha
        !=
        receipt_manifest_sha
    ):
        raise QLoRALifecycleProjectionError(
            (
                "Training manifest bytes do not "
                "match training receipt."
            )
        )


    if (
        manifest.get(
            "training_run_id"
        )
        !=
        receipt.get(
            "training_run_id"
        )
    ):
        raise QLoRALifecycleProjectionError(
            (
                "Training run identity mismatch "
                "between manifest and receipt."
            )
        )


    # ========================================================
    # ADAPTER BUNDLE AUTHORITY
    # ========================================================


    adapter = _require_mapping(
        receipt.get(
            "adapter"
        ),
        field_name="receipt.adapter",
    )


    files_raw = adapter.get(
        "files"
    )


    if not isinstance(
        files_raw,
        list,
    ):
        raise QLoRALifecycleProjectionError(
            "receipt.adapter.files must be a list."
        )


    file_count = _require_positive_int(
        adapter.get(
            "file_count"
        ),
        field_name=
            "receipt.adapter.file_count",
    )


    if (
        file_count
        !=
        len(
            files_raw
        )
    ):
        raise QLoRALifecycleProjectionError(
            (
                "Adapter file count does not "
                "match receipt file list."
            )
        )


    file_names: list[str] = []
    total_bytes = 0


    for index, raw_entry in enumerate(
        files_raw
    ):

        entry = _require_mapping(
            raw_entry,
            field_name=(
                "receipt.adapter.files"
                f"[{index}]"
            ),
        )


        name = _normalize_relative_path(
            entry.get(
                "relative_path"
            ),
            field_name=(
                "receipt.adapter.files"
                f"[{index}].relative_path"
            ),
        )


        _require_sha256(
            entry.get(
                "sha256"
            ),
            field_name=(
                "receipt.adapter.files"
                f"[{index}].sha256"
            ),
        )


        size = _require_positive_int(
            entry.get(
                "size_bytes"
            ),
            field_name=(
                "receipt.adapter.files"
                f"[{index}].size_bytes"
            ),
        )


        file_names.append(
            name
        )

        total_bytes += size


    expected_adapter_files = {
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
    }


    if (
        set(
            file_names
        )
        !=
        expected_adapter_files
    ):
        raise QLoRALifecycleProjectionError(
            (
                "Adapter receipt must describe "
                "the exact three-file PEFT bundle."
            )
        )


    if (
        len(
            file_names
        )
        !=
        len(
            set(
                file_names
            )
        )
    ):
        raise QLoRALifecycleProjectionError(
            "Duplicate adapter receipt file."
        )


    bundle_sha = _require_sha256(
        adapter.get(
            "bundle_sha256"
        ),
        field_name=
            "receipt.adapter.bundle_sha256",
    )


    # ========================================================
    # ADAPTER PATH AUTHORITY
    # ========================================================


    planned_outputs = (
        _require_mapping(
            manifest.get(
                "planned_outputs"
            ),
            field_name=
                "manifest.planned_outputs",
        )
    )


    adapter_path = (
        _normalize_relative_path(
            planned_outputs.get(
                "adapter_directory"
            ),
            field_name=(
                "manifest.planned_outputs."
                "adapter_directory"
            ),
        )
    )


    if not adapter_path.startswith(
        "artifacts/adaptation/adapters/"
    ):
        raise QLoRALifecycleProjectionError(
            (
                "QLoRA adapter path is outside "
                "the adaptation adapter namespace."
            )
        )


    # ========================================================
    # SHARED SEMANTIC CONTRACT FINGERPRINT
    # ========================================================


    semantic_contract_sha = (
        source_contract_sha256(
            contract
        )
    )


    # ========================================================
    # DETERMINISTIC SHARED ARTIFACT ID
    # ========================================================


    artifact_id = (
        "artifact:adapter:"
        +
        bundle_sha[
            :32
        ]
    )


    # ========================================================
    # NEUTRAL SHARED RECORD
    # ========================================================


    return ModelLifecycleArtifactRecord(
        artifact_id=
            artifact_id,

        artifact_family=
            "adapter",

        experiment_id=
            experiment_id,

        artifact_format=
            "peft_adapter_bundle",

        artifact_path=
            adapter_path,

        artifact_file_bytes=
            total_bytes,

        artifact_sha256=
            bundle_sha,

        created_at_utc=
            _required_text(
                receipt.get(
                    "created_at"
                ),
                field_name=
                    "receipt.created_at",
            ),

        source_contract_kind=
            contract_rule,

        source_contract_sha256=
            semantic_contract_sha,
    )
