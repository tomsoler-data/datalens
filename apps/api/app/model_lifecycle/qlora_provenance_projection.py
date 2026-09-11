from __future__ import annotations


import hashlib
import json
import re


from collections.abc import Mapping


from pathlib import Path


from typing import Any


from app.model_lifecycle.fingerprints import (
    source_contract_sha256,
)

from app.model_lifecycle.provenance_contracts import (
    ModelLifecycleEvidenceBinding,
    ModelLifecycleProvenanceRecord,
)

from app.model_lifecycle.qlora_projection import (
    QLoRALifecycleProjectionError,
    project_qlora_adapter_artifact,
)


# ============================================================
# VERSION
# ============================================================


QLORA_PROVENANCE_PROJECTION_RULE_VERSION = (
    "qlora_provenance_projection_v0.1"
)


# ============================================================
# ERROR
# ============================================================


class QLoRAProvenanceProjectionError(
    RuntimeError
):
    pass


# ============================================================
# HELPERS
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
        raise QLoRAProvenanceProjectionError(
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
        raise QLoRAProvenanceProjectionError(
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
        raise QLoRAProvenanceProjectionError(
            (
                f"{field_name} must be a "
                "64-character SHA-256 digest."
            )
        )


    return normalized


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
        raise QLoRAProvenanceProjectionError(
            f"Could not load JSON authority: {path}"
        ) from error


    if not isinstance(
        payload,
        dict,
    ):
        raise QLoRAProvenanceProjectionError(
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
        raise QLoRAProvenanceProjectionError(
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
        raise QLoRAProvenanceProjectionError(
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


def project_qlora_adapter_provenance(
    *,
    experiment_contract_path: Path,
    experiment_contract_freeze_path: Path,
    training_manifest_path: Path,
    training_receipt_path: Path,
) -> ModelLifecycleProvenanceRecord:
    """
    Project immutable QLoRA training authorities into the shared
    DataLens lifecycle provenance model.

    The projection binds metadata only. It does not load model
    weights, load PEFT, read protected evaluation cases, train,
    tune or mutate the historical adapter.
    """

    try:
        artifact = (
            project_qlora_adapter_artifact(
                experiment_contract_path=
                    experiment_contract_path,

                experiment_contract_freeze_path=
                    experiment_contract_freeze_path,

                training_manifest_path=
                    training_manifest_path,

                training_receipt_path=
                    training_receipt_path,
            )
        )

    except QLoRALifecycleProjectionError as error:
        raise QLoRAProvenanceProjectionError(
            (
                "QLoRA artifact authority "
                "projection failed."
            )
        ) from error


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
    # CORE IDENTITY
    # ========================================================


    experiment_id = _required_text(
        receipt.get(
            "experiment_id"
        ),
        field_name=
            "receipt.experiment_id",
    )


    if (
        experiment_id
        !=
        artifact.experiment_id
    ):
        raise QLoRAProvenanceProjectionError(
            (
                "Artifact and receipt "
                "experiment identities differ."
            )
        )


    execution_id = _required_text(
        receipt.get(
            "training_run_id"
        ),
        field_name=
            "receipt.training_run_id",
    )


    created_at_utc = _required_text(
        receipt.get(
            "created_at"
        ),
        field_name=
            "receipt.created_at",
    )


    # ========================================================
    # CONTRACT AUTHORITIES
    # ========================================================


    contract_rule = _required_text(
        contract.get(
            "rule_version"
        ),
        field_name=
            "contract.rule_version",
    )


    if (
        contract_rule
        !=
        artifact.source_contract_kind
    ):
        raise QLoRAProvenanceProjectionError(
            (
                "Artifact source-contract kind "
                "does not match contract."
            )
        )


    semantic_contract_sha = (
        source_contract_sha256(
            contract
        )
    )


    if (
        semantic_contract_sha
        !=
        artifact.source_contract_sha256
    ):
        raise QLoRAProvenanceProjectionError(
            (
                "Artifact source-contract "
                "fingerprint mismatch."
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


    # ========================================================
    # MANIFEST AUTHORITIES
    # ========================================================


    manifest_experiment = (
        _require_mapping(
            manifest.get(
                "experiment"
            ),
            field_name=
                "manifest.experiment",
        )
    )


    manifest_dataset = (
        _require_mapping(
            manifest.get(
                "dataset"
            ),
            field_name=
                "manifest.dataset",
        )
    )


    manifest_outputs = (
        _require_mapping(
            manifest.get(
                "planned_outputs"
            ),
            field_name=
                "manifest.planned_outputs",
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
        raise QLoRAProvenanceProjectionError(
            (
                "Manifest contract authority "
                "does not match frozen contract."
            )
        )


    freeze_sha = (
        _require_sha256(
            manifest_experiment.get(
                "experiment_contract_freeze_sha256"
            ),
            field_name=(
                "manifest.experiment."
                "experiment_contract_freeze_sha256"
            ),
        )
    )


    actual_freeze_sha = (
        _lf_normalized_sha256(
            experiment_contract_freeze_path
        )
    )


    if (
        actual_freeze_sha
        !=
        freeze_sha
    ):
        raise QLoRAProvenanceProjectionError(
            (
                "Experiment contract freeze "
                "bytes do not match manifest."
            )
        )


    # ========================================================
    # TRAINING MANIFEST
    # ========================================================


    manifest_sha = (
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
        manifest_sha
    ):
        raise QLoRAProvenanceProjectionError(
            (
                "Training manifest bytes do "
                "not match receipt."
            )
        )


    if (
        manifest.get(
            "training_run_id"
        )
        !=
        execution_id
    ):
        raise QLoRAProvenanceProjectionError(
            (
                "Manifest and receipt training "
                "run identities differ."
            )
        )


    # ========================================================
    # TRAINING RECEIPT
    # ========================================================


    if receipt.get(
        "passed"
    ) is not True:
        raise QLoRAProvenanceProjectionError(
            "Training receipt did not pass."
        )


    receipt_sha = (
        _lf_normalized_sha256(
            training_receipt_path
        )
    )


    # ========================================================
    # TRAINING DATASET
    # ========================================================


    dataset_id = _required_text(
        manifest_dataset.get(
            "dataset_id"
        ),
        field_name=
            "manifest.dataset.dataset_id",
    )


    dataset_sha = (
        _require_sha256(
            manifest_dataset.get(
                "dataset_sha256"
            ),
            field_name=
                "manifest.dataset.dataset_sha256",
        )
    )


    dataset_version = _required_text(
        manifest_dataset.get(
            "dataset_version"
        ),
        field_name=(
            "manifest.dataset.dataset_version"
        ),
    )


    # ========================================================
    # OUTPUT PATH CONSISTENCY
    # ========================================================


    planned_adapter_path = _required_text(
        manifest_outputs.get(
            "adapter_directory"
        ),
        field_name=(
            "manifest.planned_outputs."
            "adapter_directory"
        ),
    ).replace(
        "\\",
        "/",
    )


    if (
        planned_adapter_path
        !=
        artifact.artifact_path
    ):
        raise QLoRAProvenanceProjectionError(
            (
                "Artifact path differs from "
                "training manifest authority."
            )
        )


    # ========================================================
    # IMMUTABLE EVIDENCE
    # ========================================================


    evidence = (
        ModelLifecycleEvidenceBinding(
            evidence_kind=
                "source_contract_frozen",

            authority_id=(
                experiment_contract_path
                .as_posix()
            ),

            authority_sha256=
                frozen_contract_sha,
        ),

        ModelLifecycleEvidenceBinding(
            evidence_kind=
                "source_contract_freeze",

            authority_id=(
                experiment_contract_freeze_path
                .as_posix()
            ),

            authority_sha256=
                freeze_sha,
        ),

        ModelLifecycleEvidenceBinding(
            evidence_kind=
                "training_manifest",

            authority_id=(
                training_manifest_path
                .as_posix()
            ),

            authority_sha256=
                manifest_sha,
        ),

        ModelLifecycleEvidenceBinding(
            evidence_kind=
                "training_receipt",

            authority_id=(
                training_receipt_path
                .as_posix()
            ),

            authority_sha256=
                receipt_sha,
        ),

        ModelLifecycleEvidenceBinding(
            evidence_kind=
                "training_dataset",

            authority_id=(
                dataset_id
                +
                "@"
                +
                dataset_version
            ),

            authority_sha256=
                dataset_sha,
        ),
    )


    # ========================================================
    # DETERMINISTIC PROVENANCE ID
    # ========================================================


    provenance_identity = {
        "artifact_id":
            artifact.artifact_id,

        "experiment_id":
            experiment_id,

        "producer_family":
            "llm_adaptation",

        "execution_id":
            execution_id,

        "source_contract_kind":
            contract_rule,

        "source_contract_sha256":
            semantic_contract_sha,

        "evidence": [
            binding.model_dump(
                mode="json"
            )
            for binding
            in evidence
        ],
    }


    provenance_fingerprint = (
        source_contract_sha256(
            provenance_identity
        )
    )


    provenance_id = (
        "provenance:llm_adaptation:"
        +
        provenance_fingerprint[
            :32
        ]
    )


    # ========================================================
    # NEUTRAL PROVENANCE RECORD
    # ========================================================


    return ModelLifecycleProvenanceRecord(
        provenance_id=
            provenance_id,

        artifact_id=
            artifact.artifact_id,

        artifact_family=
            artifact.artifact_family,

        experiment_id=
            experiment_id,

        producer_family=
            "llm_adaptation",

        execution_id=
            execution_id,

        source_contract_kind=
            contract_rule,

        source_contract_sha256=
            semantic_contract_sha,

        evidence=
            evidence,

        created_at_utc=
            created_at_utc,
    )
