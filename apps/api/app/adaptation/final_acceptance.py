from __future__ import annotations


import hashlib
import json


from datetime import (
    datetime,
    timezone,
)

from pathlib import (
    Path,
)

from typing import (
    Literal,
    Sequence,
    Tuple,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


from app.adaptation.contracts import (
    ADAPTATION_DATA_GOVERNANCE_RULE_VERSION,
)


# ============================================================
# VERSIONS
# ============================================================


FINAL_ACCEPTANCE_HOLDOUT_CONTRACT_RULE_VERSION = (
    "final_acceptance_holdout_contract_v0.1"
)


FINAL_ACCEPTANCE_HOLDOUT_FREEZE_RULE_VERSION = (
    "final_acceptance_holdout_freeze_v0.1"
)


FINAL_ACCEPTANCE_INTEGRITY_RULE_VERSION = (
    "final_acceptance_integrity_v0.1"
)


# ============================================================
# TYPES
# ============================================================


FinalAcceptanceEvaluationRole = Literal[
    "final_acceptance_holdout"
]


FinalAcceptanceTechnicalSplit = Literal[
    "holdout"
]


FinalAcceptanceComponentRole = Literal[
    "benchmark_definition",
    "dataset_definition",
    "case_manifest",
    "supporting_definition",
]


# ============================================================
# PUBLIC MODELS
# ============================================================


class FinalAcceptanceGatePolicy(
    BaseModel
):
    """
    Acceptance gates preregistered before adaptation training.

    The numeric threshold is intentionally supplied explicitly by
    the experiment author rather than silently chosen by this module.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    minimum_normalized_score: float = Field(
        ge=0.0,
        le=1.0,
    )


    maximum_safety_failures: int = Field(
        ge=0,
    )


    maximum_dangerous_false_positives: int = Field(
        ge=0,
    )


    require_regression_gate_pass: Literal[
        True
    ] = True


    require_freeze_integrity: Literal[
        True
    ] = True


class FinalAcceptanceComponent(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    relative_path: str = Field(
        min_length=1,
    )


    role: FinalAcceptanceComponentRole


    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    size_bytes: int = Field(
        ge=1,
    )


class FinalAcceptanceHoldoutContract(
    BaseModel
):
    """
    Immutable experimental contract for the final acceptance holdout.

    This is deliberately distinct from the technical benchmark split.

    Technical mechanism:
        split = holdout

    Experimental role:
        evaluation_role = final_acceptance_holdout
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    contract_id: str = Field(
        min_length=1,
    )


    benchmark_id: str = Field(
        min_length=1,
    )


    benchmark_version: str = Field(
        min_length=1,
    )


    dataset_id: str = Field(
        min_length=1,
    )


    domain: str = Field(
        min_length=1,
    )


    technical_split: FinalAcceptanceTechnicalSplit = (
        "holdout"
    )


    evaluation_role: FinalAcceptanceEvaluationRole = (
        "final_acceptance_holdout"
    )


    independent_final_evidence: Literal[
        True
    ] = True


    frozen_before_training: Literal[
        True
    ] = True


    adaptation_tuning_input: Literal[
        False
    ] = False


    regression_answers_allowed: Literal[
        False
    ] = False


    pre_adaptation_holdout_material_allowed: Literal[
        False
    ] = False


    rag_holdout_material_allowed: Literal[
        False
    ] = False


    training_started_at_contract_creation: Literal[
        False
    ] = False


    created_at: str = Field(
        min_length=1,
    )


    column_case_count: int = Field(
        ge=0,
    )


    pair_case_count: int = Field(
        ge=0,
    )


    total_case_count: int = Field(
        ge=1,
    )


    gates: FinalAcceptanceGatePolicy


    components: Tuple[
        FinalAcceptanceComponent,
        ...,
    ] = Field(
        min_length=1,
    )


    component_count: int = Field(
        ge=1,
    )


    component_bundle_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    identity_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    governance_rule_version: Literal[
        "adaptation_data_governance_v0.1"
    ] = ADAPTATION_DATA_GOVERNANCE_RULE_VERSION


    contract_rule_version: Literal[
        "final_acceptance_holdout_contract_v0.1"
    ] = FINAL_ACCEPTANCE_HOLDOUT_CONTRACT_RULE_VERSION


    @model_validator(
        mode="after",
    )
    def validate_contract(
        self,
    ) -> "FinalAcceptanceHoldoutContract":
        expected_total = (
            self.column_case_count
            +
            self.pair_case_count
        )


        if (
            self.total_case_count
            !=
            expected_total
        ):
            raise ValueError(
                "Final acceptance total_case_count must equal "
                "column_case_count + pair_case_count."
            )


        if (
            self.component_count
            !=
            len(
                self.components
            )
        ):
            raise ValueError(
                "Final acceptance component_count mismatch."
            )


        relative_paths = [
            component.relative_path

            for component
            in self.components
        ]


        if (
            len(
                relative_paths
            )
            !=
            len(
                set(
                    relative_paths
                )
            )
        ):
            raise ValueError(
                "Final acceptance component paths "
                "must be unique."
            )


        return self


class FinalAcceptanceIntegrityMismatch(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    relative_path: str = Field(
        min_length=1,
    )


    expected_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    actual_sha256: str | None = None


    expected_size_bytes: int = Field(
        ge=1,
    )


    actual_size_bytes: int | None = Field(
        default=None,
        ge=0,
    )


    reason: Literal[
        "missing",
        "sha256_mismatch",
        "size_mismatch",
    ]


class FinalAcceptanceIntegrityReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    contract_identity_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    expected_component_count: int = Field(
        ge=1,
    )


    verified_component_count: int = Field(
        ge=0,
    )


    mismatch_count: int = Field(
        ge=0,
    )


    mismatches: Tuple[
        FinalAcceptanceIntegrityMismatch,
        ...,
    ] = ()


    current_component_bundle_sha256: str | None = None


    contract_identity_valid: bool


    component_bundle_valid: bool


    passed: bool


    rule_version: Literal[
        "final_acceptance_integrity_v0.1"
    ] = FINAL_ACCEPTANCE_INTEGRITY_RULE_VERSION


    @model_validator(
        mode="after",
    )
    def validate_report(
        self,
    ) -> "FinalAcceptanceIntegrityReport":
        if (
            self.mismatch_count
            !=
            len(
                self.mismatches
            )
        ):
            raise ValueError(
                "Final acceptance mismatch_count mismatch."
            )


        expected_pass = (
            self.contract_identity_valid
            and
            self.component_bundle_valid
            and
            self.mismatch_count
            ==
            0
            and
            self.verified_component_count
            ==
            self.expected_component_count
        )


        if (
            self.passed
            !=
            expected_pass
        ):
            raise ValueError(
                "Final acceptance integrity status "
                "is inconsistent."
            )


        return self


class FinalAcceptanceFreezeArtifact(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    status: Literal[
        "frozen"
    ] = "frozen"


    evaluation_role: FinalAcceptanceEvaluationRole = (
        "final_acceptance_holdout"
    )


    technical_split: FinalAcceptanceTechnicalSplit = (
        "holdout"
    )


    frozen_before_training: Literal[
        True
    ] = True


    adaptation_tuning_input: Literal[
        False
    ] = False


    training_started_at_freeze: Literal[
        False
    ] = False


    contract: FinalAcceptanceHoldoutContract


    contract_identity_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    component_bundle_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    frozen_at: str = Field(
        min_length=1,
    )


    freeze_rule_version: Literal[
        "final_acceptance_holdout_freeze_v0.1"
    ] = FINAL_ACCEPTANCE_HOLDOUT_FREEZE_RULE_VERSION


    @model_validator(
        mode="after",
    )
    def validate_freeze(
        self,
    ) -> "FinalAcceptanceFreezeArtifact":
        if (
            self.contract_identity_sha256
            !=
            self.contract.identity_sha256
        ):
            raise ValueError(
                "Freeze artifact contract identity mismatch."
            )


        if (
            self.component_bundle_sha256
            !=
            self.contract.component_bundle_sha256
        ):
            raise ValueError(
                "Freeze artifact component bundle mismatch."
            )


        return self


# ============================================================
# HASHING
# ============================================================


def _sha256_bytes(
    value: bytes,
) -> str:
    return hashlib.sha256(
        value
    ).hexdigest()


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()


    with path.open(
        "rb",
    ) as handle:
        while True:
            chunk = handle.read(
                8
                *
                1024
                *
                1024
            )


            if not chunk:
                break


            digest.update(
                chunk
            )


    return digest.hexdigest()


def _canonical_json_bytes(
    value: object,
) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
    ).encode(
        "utf-8"
    )


def _canonical_sha256(
    value: object,
) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            value
        )
    )


# ============================================================
# TIME
# ============================================================


def _utc_now_iso8601() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .replace(
            microsecond=0
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


# ============================================================
# PATHS
# ============================================================


def _repository_relative_path(
    *,
    repository_root: Path,
    path: Path,
) -> str:
    repository_root = (
        repository_root
        .expanduser()
        .resolve()
    )


    path = (
        path
        .expanduser()
        .resolve()
    )


    try:
        relative = path.relative_to(
            repository_root
        )

    except ValueError as error:
        raise ValueError(
            "Final acceptance components must live "
            "inside the repository."
        ) from error


    return relative.as_posix()


# ============================================================
# COMPONENTS
# ============================================================


def _build_component(
    *,
    repository_root: Path,
    path: Path,
    role: FinalAcceptanceComponentRole,
) -> FinalAcceptanceComponent:
    path = (
        path
        .expanduser()
        .resolve()
    )


    if not path.is_file():
        raise FileNotFoundError(
            "Final acceptance component is missing: "
            f"{path}"
        )


    size_bytes = (
        path.stat()
        .st_size
    )


    if (
        size_bytes
        <=
        0
    ):
        raise ValueError(
            "Final acceptance components may not "
            "be empty."
        )


    return FinalAcceptanceComponent(
        relative_path=
            _repository_relative_path(
                repository_root=
                    repository_root,
                path=path,
            ),
        role=
            role,
        sha256=
            _sha256_file(
                path
            ),
        size_bytes=
            size_bytes,
    )


def _component_bundle_payload(
    components: Sequence[
        FinalAcceptanceComponent
    ],
) -> list[
    dict[
        str,
        object,
    ]
]:
    return [
        {
            "relative_path":
                component.relative_path,
            "role":
                component.role,
            "sha256":
                component.sha256,
            "size_bytes":
                component.size_bytes,
        }

        for component
        in sorted(
            components,
            key=lambda item: (
                item.relative_path,
                item.role,
            ),
        )
    ]


def _component_bundle_sha256(
    components: Sequence[
        FinalAcceptanceComponent
    ],
) -> str:
    return _canonical_sha256(
        _component_bundle_payload(
            components
        )
    )


# ============================================================
# CONTRACT IDENTITY
# ============================================================


def _contract_identity_payload(
    *,
    contract_id: str,
    benchmark_id: str,
    benchmark_version: str,
    dataset_id: str,
    domain: str,
    created_at: str,
    column_case_count: int,
    pair_case_count: int,
    total_case_count: int,
    gates: FinalAcceptanceGatePolicy,
    components: Sequence[
        FinalAcceptanceComponent
    ],
    component_bundle_sha256: str,
) -> dict[
    str,
    object,
]:
    return {
        "contract_id":
            contract_id,
        "benchmark_id":
            benchmark_id,
        "benchmark_version":
            benchmark_version,
        "dataset_id":
            dataset_id,
        "domain":
            domain,
        "technical_split":
            "holdout",
        "evaluation_role":
            "final_acceptance_holdout",
        "independent_final_evidence":
            True,
        "frozen_before_training":
            True,
        "adaptation_tuning_input":
            False,
        "regression_answers_allowed":
            False,
        "pre_adaptation_holdout_material_allowed":
            False,
        "rag_holdout_material_allowed":
            False,
        "training_started_at_contract_creation":
            False,
        "created_at":
            created_at,
        "column_case_count":
            column_case_count,
        "pair_case_count":
            pair_case_count,
        "total_case_count":
            total_case_count,
        "gates":
            gates.model_dump(
                mode="json"
            ),
        "components":
            [
                component.model_dump(
                    mode="json"
                )

                for component
                in components
            ],
        "component_count":
            len(
                components
            ),
        "component_bundle_sha256":
            component_bundle_sha256,
        "governance_rule_version":
            ADAPTATION_DATA_GOVERNANCE_RULE_VERSION,
        "contract_rule_version":
            FINAL_ACCEPTANCE_HOLDOUT_CONTRACT_RULE_VERSION,
    }


def compute_contract_identity_sha256(
    contract: FinalAcceptanceHoldoutContract,
) -> str:
    payload = (
        _contract_identity_payload(
            contract_id=
                contract.contract_id,
            benchmark_id=
                contract.benchmark_id,
            benchmark_version=
                contract.benchmark_version,
            dataset_id=
                contract.dataset_id,
            domain=
                contract.domain,
            created_at=
                contract.created_at,
            column_case_count=
                contract.column_case_count,
            pair_case_count=
                contract.pair_case_count,
            total_case_count=
                contract.total_case_count,
            gates=
                contract.gates,
            components=
                contract.components,
            component_bundle_sha256=
                contract.component_bundle_sha256,
        )
    )


    return _canonical_sha256(
        payload
    )


# ============================================================
# CONTRACT BUILDER
# ============================================================


def build_final_acceptance_holdout_contract(
    *,
    repository_root: Path,
    contract_id: str,
    benchmark_id: str,
    benchmark_version: str,
    dataset_id: str,
    domain: str,
    column_case_count: int,
    pair_case_count: int,
    gates: FinalAcceptanceGatePolicy,
    component_paths: Sequence[
        tuple[
            Path,
            FinalAcceptanceComponentRole,
        ]
    ],
    training_has_started: bool = False,
    created_at: str | None = None,
) -> FinalAcceptanceHoldoutContract:
    if training_has_started:
        raise RuntimeError(
            "Final acceptance holdout contract must "
            "be created before adaptation training."
        )


    if not contract_id.strip():
        raise ValueError(
            "contract_id is required."
        )


    if not benchmark_id.strip():
        raise ValueError(
            "benchmark_id is required."
        )


    if not benchmark_version.strip():
        raise ValueError(
            "benchmark_version is required."
        )


    if not dataset_id.strip():
        raise ValueError(
            "dataset_id is required."
        )


    if not domain.strip():
        raise ValueError(
            "domain is required."
        )


    if (
        column_case_count
        <
        0
        or
        pair_case_count
        <
        0
    ):
        raise ValueError(
            "Final acceptance case counts "
            "may not be negative."
        )


    total_case_count = (
        column_case_count
        +
        pair_case_count
    )


    if (
        total_case_count
        <=
        0
    ):
        raise ValueError(
            "Final acceptance holdout must contain "
            "at least one case."
        )


    if not component_paths:
        raise ValueError(
            "At least one final acceptance component "
            "is required."
        )


    repository_root = (
        repository_root
        .expanduser()
        .resolve()
    )


    if not repository_root.is_dir():
        raise FileNotFoundError(
            "Repository root does not exist."
        )


    components = tuple(
        _build_component(
            repository_root=
                repository_root,
            path=path,
            role=role,
        )

        for (
            path,
            role,
        )
        in component_paths
    )


    relative_paths = [
        component.relative_path

        for component
        in components
    ]


    if (
        len(
            relative_paths
        )
        !=
        len(
            set(
                relative_paths
            )
        )
    ):
        raise ValueError(
            "Duplicate final acceptance component "
            "path detected."
        )


    bundle_sha256 = (
        _component_bundle_sha256(
            components
        )
    )


    timestamp = (
        created_at
        if created_at is not None
        else
        _utc_now_iso8601()
    )


    identity_payload = (
        _contract_identity_payload(
            contract_id=
                contract_id.strip(),
            benchmark_id=
                benchmark_id.strip(),
            benchmark_version=
                benchmark_version.strip(),
            dataset_id=
                dataset_id.strip(),
            domain=
                domain.strip(),
            created_at=
                timestamp,
            column_case_count=
                column_case_count,
            pair_case_count=
                pair_case_count,
            total_case_count=
                total_case_count,
            gates=
                gates,
            components=
                components,
            component_bundle_sha256=
                bundle_sha256,
        )
    )


    identity_sha256 = (
        _canonical_sha256(
            identity_payload
        )
    )


    return FinalAcceptanceHoldoutContract(
        contract_id=
            contract_id.strip(),
        benchmark_id=
            benchmark_id.strip(),
        benchmark_version=
            benchmark_version.strip(),
        dataset_id=
            dataset_id.strip(),
        domain=
            domain.strip(),
        created_at=
            timestamp,
        column_case_count=
            column_case_count,
        pair_case_count=
            pair_case_count,
        total_case_count=
            total_case_count,
        gates=
            gates,
        components=
            components,
        component_count=
            len(
                components
            ),
        component_bundle_sha256=
            bundle_sha256,
        identity_sha256=
            identity_sha256,
    )


# ============================================================
# INTEGRITY VERIFICATION
# ============================================================


def verify_final_acceptance_holdout_contract(
    *,
    repository_root: Path,
    contract: FinalAcceptanceHoldoutContract,
) -> FinalAcceptanceIntegrityReport:
    repository_root = (
        repository_root
        .expanduser()
        .resolve()
    )


    expected_identity = (
        compute_contract_identity_sha256(
            contract
        )
    )


    identity_valid = (
        expected_identity
        ==
        contract.identity_sha256
    )


    mismatches = []


    current_components = []


    verified_component_count = 0


    for component in contract.components:
        path = (
            repository_root
            /
            Path(
                component.relative_path
            )
        )


        if not path.is_file():
            mismatches.append(
                FinalAcceptanceIntegrityMismatch(
                    relative_path=
                        component.relative_path,
                    expected_sha256=
                        component.sha256,
                    actual_sha256=None,
                    expected_size_bytes=
                        component.size_bytes,
                    actual_size_bytes=None,
                    reason=
                        "missing",
                )
            )


            continue


        actual_size = (
            path.stat()
            .st_size
        )


        actual_sha256 = (
            _sha256_file(
                path
            )
        )


        if (
            actual_size
            !=
            component.size_bytes
        ):
            mismatches.append(
                FinalAcceptanceIntegrityMismatch(
                    relative_path=
                        component.relative_path,
                    expected_sha256=
                        component.sha256,
                    actual_sha256=
                        actual_sha256,
                    expected_size_bytes=
                        component.size_bytes,
                    actual_size_bytes=
                        actual_size,
                    reason=
                        "size_mismatch",
                )
            )


            continue


        if (
            actual_sha256
            !=
            component.sha256
        ):
            mismatches.append(
                FinalAcceptanceIntegrityMismatch(
                    relative_path=
                        component.relative_path,
                    expected_sha256=
                        component.sha256,
                    actual_sha256=
                        actual_sha256,
                    expected_size_bytes=
                        component.size_bytes,
                    actual_size_bytes=
                        actual_size,
                    reason=
                        "sha256_mismatch",
                )
            )


            continue


        verified_component_count += 1


        current_components.append(
            FinalAcceptanceComponent(
                relative_path=
                    component.relative_path,
                role=
                    component.role,
                sha256=
                    actual_sha256,
                size_bytes=
                    actual_size,
            )
        )


    current_bundle_sha256 = None


    if (
        len(
            current_components
        )
        ==
        len(
            contract.components
        )
    ):
        current_bundle_sha256 = (
            _component_bundle_sha256(
                current_components
            )
        )


    component_bundle_valid = (
        current_bundle_sha256
        is not None
        and
        current_bundle_sha256
        ==
        contract.component_bundle_sha256
    )


    passed = (
        identity_valid
        and
        component_bundle_valid
        and
        not mismatches
        and
        verified_component_count
        ==
        len(
            contract.components
        )
    )


    return FinalAcceptanceIntegrityReport(
        contract_identity_sha256=
            contract.identity_sha256,
        expected_component_count=
            len(
                contract.components
            ),
        verified_component_count=
            verified_component_count,
        mismatch_count=
            len(
                mismatches
            ),
        mismatches=
            tuple(
                mismatches
            ),
        current_component_bundle_sha256=
            current_bundle_sha256,
        contract_identity_valid=
            identity_valid,
        component_bundle_valid=
            component_bundle_valid,
        passed=
            passed,
    )


def assert_final_acceptance_integrity(
    report: FinalAcceptanceIntegrityReport,
) -> None:
    if not report.passed:
        raise RuntimeError(
            "Final acceptance holdout integrity gate failed. "
            f"Mismatches={report.mismatch_count}; "
            "contract_identity_valid="
            f"{report.contract_identity_valid}; "
            "component_bundle_valid="
            f"{report.component_bundle_valid}."
        )


# ============================================================
# FREEZE
# ============================================================


def build_final_acceptance_freeze_artifact(
    *,
    repository_root: Path,
    contract: FinalAcceptanceHoldoutContract,
    training_has_started: bool = False,
    frozen_at: str | None = None,
) -> FinalAcceptanceFreezeArtifact:
    if training_has_started:
        raise RuntimeError(
            "Final acceptance holdout cannot be frozen "
            "after adaptation training has started."
        )


    integrity = (
        verify_final_acceptance_holdout_contract(
            repository_root=
                repository_root,
            contract=
                contract,
        )
    )


    assert_final_acceptance_integrity(
        integrity
    )


    return FinalAcceptanceFreezeArtifact(
        contract=
            contract,
        contract_identity_sha256=
            contract.identity_sha256,
        component_bundle_sha256=
            contract.component_bundle_sha256,
        frozen_at=(
            frozen_at
            if frozen_at is not None
            else
            _utc_now_iso8601()
        ),
    )


def write_final_acceptance_freeze(
    *,
    artifact: FinalAcceptanceFreezeArtifact,
    output_path: Path,
) -> str:
    output_path = (
        output_path
        .expanduser()
        .resolve()
    )


    if output_path.exists():
        raise FileExistsError(
            "Final acceptance freeze artifact "
            "already exists: "
            f"{output_path}"
        )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    serialized = json.dumps(
        artifact.model_dump(
            mode="json"
        ),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
    )


    output_path.write_text(
        serialized
        +
        "\n",
        encoding="utf-8",
        newline="\n",
    )


    return _sha256_file(
        output_path
    )


def load_final_acceptance_freeze(
    path: Path,
) -> FinalAcceptanceFreezeArtifact:
    path = (
        path
        .expanduser()
        .resolve()
    )


    if not path.is_file():
        raise FileNotFoundError(
            "Final acceptance freeze artifact "
            f"is missing: {path}"
        )


    payload = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


    return FinalAcceptanceFreezeArtifact.model_validate(
        payload
    )


def assert_final_acceptance_frozen_before_training(
    artifact: FinalAcceptanceFreezeArtifact,
) -> None:
    if (
        artifact.status
        !=
        "frozen"
    ):
        raise RuntimeError(
            "Final acceptance holdout is not frozen."
        )


    if not artifact.frozen_before_training:
        raise RuntimeError(
            "Final acceptance holdout was not frozen "
            "before training."
        )


    if artifact.adaptation_tuning_input:
        raise RuntimeError(
            "Final acceptance holdout may not be used "
            "as adaptation tuning input."
        )


    if artifact.training_started_at_freeze:
        raise RuntimeError(
            "Training had already started when the "
            "final acceptance holdout was frozen."
        )


    if (
        artifact.contract.evaluation_role
        !=
        "final_acceptance_holdout"
    ):
        raise RuntimeError(
            "Unexpected final acceptance evaluation role."
        )


    if (
        artifact.contract.technical_split
        !=
        "holdout"
    ):
        raise RuntimeError(
            "Final acceptance benchmark must use "
            "the native holdout split."
        )
