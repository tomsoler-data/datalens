from __future__ import annotations


import hashlib
import json
import re


from pathlib import Path


from typing import Literal


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.model_lifecycle.registry import (
    ModelLifecycleRegistryError,
    ModelLifecycleRegistryNotFoundError,
    get_model_lifecycle_entry,
)


# ============================================================
# VERSION
# ============================================================


TRUSTED_ARTIFACT_RESOLVER_RULE_VERSION = (
    "trusted_artifact_resolver_v0.1"
)


# ============================================================
# CONSTANTS
# ============================================================


PEFT_ADAPTER_FILES = (
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
)


SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)


# ============================================================
# ERROR
# ============================================================


class TrustedArtifactResolutionError(
    RuntimeError
):
    pass


# ============================================================
# RESULT CONTRACTS
# ============================================================


class TrustedArtifactFileRecord(
    BaseModel
):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    relative_path: str = Field(
        min_length=1,
    )


    physical_path: str = Field(
        min_length=1,
    )


    size_bytes: int = Field(
        gt=0,
    )


    sha256: str = Field(
        min_length=64,
        max_length=64,
    )


class TrustedPEFTAdapterResolution(
    BaseModel
):
    """
    Verified physical resolution of one registered PEFT adapter.

    This is integrity metadata only. It does not instantiate,
    deserialize or load a model or adapter.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    artifact_id: str = Field(
        min_length=1,
    )


    provenance_id: str = Field(
        min_length=1,
    )


    experiment_id: str = Field(
        min_length=1,
    )


    artifact_path: str = Field(
        min_length=1,
    )


    physical_adapter_path: str = Field(
        min_length=1,
    )


    artifact_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    artifact_file_bytes: int = Field(
        gt=0,
    )


    training_receipt_path: str = Field(
        min_length=1,
    )


    training_receipt_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    files: tuple[
        TrustedArtifactFileRecord,
        ...,
    ]


    rule_version: Literal[
        "trusted_artifact_resolver_v0.1"
    ] = (
        TRUSTED_ARTIFACT_RESOLVER_RULE_VERSION
    )


# ============================================================
# HELPERS
# ============================================================


def _api_root(
) -> Path:

    return (
        Path(
            __file__
        )
        .resolve()
        .parents[
            2
        ]
    )


def _required_sha256(
    value: object,
    *,
    field_name: str,
) -> str:

    normalized = str(
        value
        if value is not None
        else ""
    ).strip().lower()


    if (
        SHA256_PATTERN.fullmatch(
            normalized
        )
        is None
    ):
        raise TrustedArtifactResolutionError(
            (
                f"{field_name} must be a "
                "64-character SHA-256 digest."
            )
        )


    return normalized


def _sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()


    try:

        with path.open(
            "rb"
        ) as handle:

            for chunk in iter(
                lambda:
                    handle.read(
                        8 * 1024 * 1024
                    ),
                b"",
            ):

                digest.update(
                    chunk
                )


    except OSError as error:
        raise TrustedArtifactResolutionError(
            (
                "Could not read artifact file: "
                f"{path}"
            )
        ) from error


    return digest.hexdigest()


def _lf_normalized_text_sha256(
    path: Path,
) -> str:

    try:
        raw = path.read_bytes()

    except OSError as error:
        raise TrustedArtifactResolutionError(
            (
                "Could not read text authority: "
                f"{path}"
            )
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
        raise TrustedArtifactResolutionError(
            (
                "Text authority is not UTF-8: "
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


def _load_json_object(
    path: Path,
) -> dict[str, object]:

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
        raise TrustedArtifactResolutionError(
            (
                "Could not load trusted JSON "
                f"authority: {path}"
            )
        ) from error


    if not isinstance(
        payload,
        dict,
    ):
        raise TrustedArtifactResolutionError(
            (
                "Trusted JSON authority must "
                f"be an object: {path}"
            )
        )


    return payload


def _resolve_receipt_path(
    authority_id: str,
    *,
    api_root: Path,
) -> Path:

    raw = Path(
        authority_id
    )


    receipt_root = (
        api_root
        /
        "artifacts"
        /
        "adaptation"
        /
        "training"
    ).resolve()


    candidates: list[
        Path
    ] = []


    if raw.is_absolute():

        candidates.append(
            raw.resolve()
        )


    else:

        candidates.append(
            (
                Path.cwd()
                /
                raw
            ).resolve()
        )

        candidates.append(
            (
                api_root
                /
                raw
            ).resolve()
        )


    seen: set[
        str
    ] = set()


    for candidate in candidates:

        key = str(
            candidate
        )


        if key in seen:
            continue


        seen.add(
            key
        )


        if (
            candidate.is_file()
            and
            candidate.is_relative_to(
                receipt_root
            )
        ):
            return candidate


    raise TrustedArtifactResolutionError(
        (
            "Training receipt evidence does not "
            "resolve beneath the trusted adaptation "
            "training root."
        )
    )


# ============================================================
# TRUSTED RESOLUTION
# ============================================================


def resolve_trusted_peft_adapter_artifact(
    artifact_id: str,
    *,
    registry_path: Path | str | None = None,
    api_root: Path | str | None = None,
) -> TrustedPEFTAdapterResolution:
    """
    Resolve one lifecycle artifact to a physically verified PEFT
    adapter bundle.

    The resolver reads bytes only to establish integrity. It does
    not import torch/transformers/peft, instantiate a model,
    deserialize tensors or execute inference.
    """

    root = (
        Path(
            api_root
        ).resolve()

        if api_root is not None

        else _api_root()
    )


    try:

        entry = (
            get_model_lifecycle_entry(
                artifact_id,
                registry_path=
                    registry_path,
            )
        )


    except (
        ModelLifecycleRegistryError,
        ModelLifecycleRegistryNotFoundError,
    ) as error:
        raise TrustedArtifactResolutionError(
            (
                "Could not resolve lifecycle "
                f"artifact: {artifact_id}"
            )
        ) from error


    artifact = (
        entry.artifact
    )

    provenance = (
        entry.provenance
    )


    # ========================================================
    # REGISTERED TYPE AUTHORITY
    # ========================================================


    if (
        artifact.artifact_family
        !=
        "adapter"
    ):
        raise TrustedArtifactResolutionError(
            (
                "Trusted PEFT resolver requires "
                "artifact_family='adapter'."
            )
        )


    if (
        artifact.artifact_format
        !=
        "peft_adapter_bundle"
    ):
        raise TrustedArtifactResolutionError(
            (
                "Trusted PEFT resolver requires "
                "artifact_format="
                "'peft_adapter_bundle'."
            )
        )


    if (
        provenance.producer_family
        !=
        "llm_adaptation"
    ):
        raise TrustedArtifactResolutionError(
            (
                "Trusted PEFT resolver requires "
                "producer_family='llm_adaptation'."
            )
        )


    # ========================================================
    # PHYSICAL PATH AUTHORITY
    # ========================================================


    raw_artifact_path = Path(
        artifact.artifact_path
    )


    if raw_artifact_path.is_absolute():
        raise TrustedArtifactResolutionError(
            (
                "Lifecycle artifact_path must "
                "remain API-relative."
            )
        )


    adapter_root = (
        root
        /
        "artifacts"
        /
        "adaptation"
        /
        "adapters"
    ).resolve()


    physical_adapter = (
        root
        /
        raw_artifact_path
    ).resolve()


    if (
        not physical_adapter.is_relative_to(
            adapter_root
        )
    ):
        raise TrustedArtifactResolutionError(
            (
                "Resolved adapter escaped the "
                "trusted adaptation adapter root."
            )
        )


    if not physical_adapter.is_dir():
        raise TrustedArtifactResolutionError(
            (
                "Resolved adapter directory "
                "does not exist."
            )
        )


    physical_entries = {
        path.name:
            path

        for path
        in physical_adapter.iterdir()
    }


    if (
        set(
            physical_entries
        )
        !=
        set(
            PEFT_ADAPTER_FILES
        )
    ):
        raise TrustedArtifactResolutionError(
            (
                "PEFT adapter directory must contain "
                "exactly the trusted three-file bundle."
            )
        )


    for name in PEFT_ADAPTER_FILES:

        path = physical_entries[
            name
        ]


        if (
            not path.is_file()
            or
            path.is_symlink()
        ):
            raise TrustedArtifactResolutionError(
                (
                    "Trusted adapter component must "
                    f"be a regular file: {name}"
                )
            )


    # ========================================================
    # TRAINING RECEIPT EVIDENCE
    # ========================================================


    receipt_bindings = [
        binding

        for binding
        in provenance.evidence

        if (
            binding.evidence_kind
            ==
            "training_receipt"
        )
    ]


    if len(
        receipt_bindings
    ) != 1:
        raise TrustedArtifactResolutionError(
            (
                "Exactly one training_receipt "
                "evidence binding is required."
            )
        )


    receipt_binding = (
        receipt_bindings[
            0
        ]
    )


    receipt_path = (
        _resolve_receipt_path(
            receipt_binding.authority_id,
            api_root=
                root,
        )
    )


    expected_receipt_sha = (
        _required_sha256(
            receipt_binding.authority_sha256,
            field_name=
                "training_receipt.authority_sha256",
        )
    )


    actual_receipt_sha = (
        _lf_normalized_text_sha256(
            receipt_path
        )
    )


    if (
        actual_receipt_sha
        !=
        expected_receipt_sha
    ):
        raise TrustedArtifactResolutionError(
            (
                "Training receipt bytes do not "
                "match provenance authority."
            )
        )


    receipt = (
        _load_json_object(
            receipt_path
        )
    )


    # ========================================================
    # RECEIPT IDENTITY
    # ========================================================


    if receipt.get(
        "passed"
    ) is not True:
        raise TrustedArtifactResolutionError(
            (
                "Training receipt is not a "
                "successful training authority."
            )
        )


    if (
        receipt.get(
            "experiment_id"
        )
        !=
        artifact.experiment_id
    ):
        raise TrustedArtifactResolutionError(
            (
                "Training receipt experiment_id "
                "does not match lifecycle artifact."
            )
        )


    if (
        receipt.get(
            "training_run_id"
        )
        !=
        provenance.execution_id
    ):
        raise TrustedArtifactResolutionError(
            (
                "Training receipt execution identity "
                "does not match provenance."
            )
        )


    # ========================================================
    # RECEIPT ADAPTER BUNDLE
    # ========================================================


    adapter_receipt = (
        receipt.get(
            "adapter"
        )
    )


    if not isinstance(
        adapter_receipt,
        dict,
    ):
        raise TrustedArtifactResolutionError(
            (
                "Training receipt adapter authority "
                "must be an object."
            )
        )


    receipt_bundle_sha = (
        _required_sha256(
            adapter_receipt.get(
                "bundle_sha256"
            ),
            field_name=
                "receipt.adapter.bundle_sha256",
        )
    )


    if (
        receipt_bundle_sha
        !=
        artifact.artifact_sha256
    ):
        raise TrustedArtifactResolutionError(
            (
                "Training receipt bundle SHA does "
                "not match lifecycle Artifact SHA."
            )
        )


    files_raw = adapter_receipt.get(
        "files"
    )


    if not isinstance(
        files_raw,
        list,
    ):
        raise TrustedArtifactResolutionError(
            (
                "Training receipt adapter files "
                "must be a list."
            )
        )


    if (
        adapter_receipt.get(
            "file_count"
        )
        !=
        len(
            files_raw
        )
    ):
        raise TrustedArtifactResolutionError(
            (
                "Training receipt adapter file_count "
                "does not match file entries."
            )
        )


    if len(
        files_raw
    ) != len(
        PEFT_ADAPTER_FILES
    ):
        raise TrustedArtifactResolutionError(
            (
                "Training receipt must describe "
                "exactly three PEFT adapter files."
            )
        )


    receipt_files: dict[
        str,
        dict[str, object],
    ] = {}


    for raw_entry in files_raw:

        if not isinstance(
            raw_entry,
            dict,
        ):
            raise TrustedArtifactResolutionError(
                (
                    "Training receipt file entry "
                    "must be an object."
                )
            )


        name = str(
            raw_entry.get(
                "relative_path",
                ""
            )
        ).strip()


        if (
            name
            not in
            PEFT_ADAPTER_FILES
        ):
            raise TrustedArtifactResolutionError(
                (
                    "Training receipt contains "
                    f"unexpected adapter file: {name!r}"
                )
            )


        if name in receipt_files:
            raise TrustedArtifactResolutionError(
                (
                    "Training receipt contains "
                    f"duplicate adapter file: {name}"
                )
            )


        receipt_files[
            name
        ] = raw_entry


    if (
        set(
            receipt_files
        )
        !=
        set(
            PEFT_ADAPTER_FILES
        )
    ):
        raise TrustedArtifactResolutionError(
            (
                "Training receipt adapter file set "
                "does not match trusted bundle."
            )
        )


    # ========================================================
    # PHYSICAL BYTE INTEGRITY
    # ========================================================


    verified_files: list[
        TrustedArtifactFileRecord
    ] = []


    total_bytes = 0


    for name in PEFT_ADAPTER_FILES:

        receipt_file = (
            receipt_files[
                name
            ]
        )


        expected_sha = (
            _required_sha256(
                receipt_file.get(
                    "sha256"
                ),
                field_name=(
                    "receipt.adapter.files."
                    f"{name}.sha256"
                ),
            )
        )


        expected_bytes_raw = (
            receipt_file.get(
                "size_bytes"
            )
        )


        if (
            isinstance(
                expected_bytes_raw,
                bool,
            )
            or
            not isinstance(
                expected_bytes_raw,
                int,
            )
            or
            expected_bytes_raw <= 0
        ):
            raise TrustedArtifactResolutionError(
                (
                    "Invalid receipt size authority "
                    f"for {name}."
                )
            )


        path = (
            physical_entries[
                name
            ]
        )


        actual_bytes = (
            path.stat().st_size
        )


        if (
            actual_bytes
            !=
            expected_bytes_raw
        ):
            raise TrustedArtifactResolutionError(
                (
                    "Physical adapter byte size "
                    f"mismatch: {name}"
                )
            )


        actual_sha = (
            _sha256_file(
                path
            )
        )


        if (
            actual_sha
            !=
            expected_sha
        ):
            raise TrustedArtifactResolutionError(
                (
                    "Physical adapter SHA-256 "
                    f"mismatch: {name}"
                )
            )


        total_bytes += (
            actual_bytes
        )


        verified_files.append(
            TrustedArtifactFileRecord(
                relative_path=
                    name,

                physical_path=
                    str(
                        path.resolve()
                    ),

                size_bytes=
                    actual_bytes,

                sha256=
                    actual_sha,
            )
        )


    if (
        total_bytes
        !=
        artifact.artifact_file_bytes
    ):
        raise TrustedArtifactResolutionError(
            (
                "Verified adapter total bytes do "
                "not match lifecycle Artifact."
            )
        )


    # ========================================================
    # TRUSTED RESULT
    # ========================================================


    return TrustedPEFTAdapterResolution(
        artifact_id=
            artifact.artifact_id,

        provenance_id=
            provenance.provenance_id,

        experiment_id=
            artifact.experiment_id,

        artifact_path=
            artifact.artifact_path,

        physical_adapter_path=
            str(
                physical_adapter
            ),

        artifact_sha256=
            artifact.artifact_sha256,

        artifact_file_bytes=
            artifact.artifact_file_bytes,

        training_receipt_path=
            str(
                receipt_path
            ),

        training_receipt_sha256=
            actual_receipt_sha,

        files=
            tuple(
                verified_files
            ),
    )
