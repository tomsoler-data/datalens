from __future__ import annotations


from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# VERSIONS
# ============================================================


QLORA_EXPERIMENT_CONTRACT_RULE_VERSION = (
    "qlora_experiment_contract_v0.1"
)


ADAPTATION_DATA_GOVERNANCE_RULE_VERSION = (
    "adaptation_data_governance_v0.1"
)


QLORA_TARGET_RESOLVER_RULE_VERSION = (
    "qlora_target_resolver_v0.1"
)


# ============================================================
# TYPES
# ============================================================


AdaptationMethod = Literal[
    "qlora",
]


QuantizationType = Literal[
    "nf4",
]


ComputeDtype = Literal[
    "bfloat16",
]


LoRATaskType = Literal[
    "CAUSAL_LM",
]


LoRABiasPolicy = Literal[
    "none",
]


LoRATargetStrategy = Literal[
    "language_model_all_linear",
]


AdaptationDatasetScope = Literal[
    "adaptation_training",
]


AdaptationEvidencePhase = Literal[
    "regression_baseline",
    "pre_adaptation_holdout",
    "final_acceptance_holdout_freeze",
]


OptimizerName = Literal[
    "paged_adamw_8bit",
]


SchedulerName = Literal[
    "cosine",
]


# ============================================================
# HELPERS
# ============================================================


_SHA256_PATTERN = (
    r"^[0-9a-f]{64}$"
)


_GIT_SHA_PATTERN = (
    r"^[0-9a-f]{40}$"
)


# ============================================================
# BASE MODEL
# ============================================================


class BaseModelReference(
    BaseModel
):
    """
    Immutable reference to the exact upstream model used by an
    adaptation experiment.

    A floating branch such as "main" is intentionally forbidden.

    DataLens must be able to reconstruct which exact model revision
    was used after the upstream repository has changed.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    repository: str = Field(
        min_length=1,
    )


    revision: str = Field(
        pattern=_GIT_SHA_PATTERN,
    )


    tokenizer_revision: str = Field(
        pattern=_GIT_SHA_PATTERN,
    )


    model_family: str = Field(
        min_length=1,
    )


    modality_scope: Literal[
        "text_only",
    ] = "text_only"


    trust_remote_code: Literal[
        False,
    ] = False


    use_multimodal_inputs: Literal[
        False,
    ] = False


    @model_validator(
        mode="after",
    )
    def validate_revision_alignment(
        self,
    ) -> "BaseModelReference":
        if (
            self.revision
            !=
            self.tokenizer_revision
        ):
            raise ValueError(
                "Base model and tokenizer revisions must match "
                "for QLoRA experiment contract v0.1."
            )


        return self


# ============================================================
# QUANTIZATION
# ============================================================


class QLoRAQuantizationPolicy(
    BaseModel
):
    """
    Frozen 4-bit loading policy for the base model.

    Only adapter parameters may become trainable later.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    load_in_4bit: Literal[
        True,
    ] = True


    quantization_type: QuantizationType = (
        "nf4"
    )


    use_double_quantization: Literal[
        True,
    ] = True


    compute_dtype: ComputeDtype = (
        "bfloat16"
    )


# ============================================================
# LORA
# ============================================================


class QLoRAParameters(
    BaseModel
):
    """
    Trainable-adapter policy.

    v0.1 deliberately specifies a semantic target strategy rather
    than accepting arbitrary caller-controlled module names.

    A separate server-owned resolver must later map
    language_model_all_linear onto the concrete language-model
    modules of the pinned architecture.

    This prevents accidentally adapting the Gemma vision tower.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    rank: int = Field(
        default=16,
        ge=1,
        le=256,
    )


    alpha: int = Field(
        default=32,
        ge=1,
        le=1024,
    )


    dropout: float = Field(
        default=0.05,
        ge=0.0,
        lt=1.0,
    )


    bias: LoRABiasPolicy = (
        "none"
    )


    task_type: LoRATaskType = (
        "CAUSAL_LM"
    )


    target_strategy: LoRATargetStrategy = (
        "language_model_all_linear"
    )


    target_resolver_rule_version: Literal[
        "qlora_target_resolver_v0.1"
    ] = QLORA_TARGET_RESOLVER_RULE_VERSION


# ============================================================
# TRAINING POLICY
# ============================================================


class AdaptationTrainingPolicy(
    BaseModel
):
    """
    Reproducible training configuration selected before model
    adaptation begins.

    Values are intentionally conservative for the validated
    RTX 4060 8 GiB laboratory environment.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    random_seed: int = Field(
        default=42,
        ge=0,
        le=2_147_483_647,
    )


    max_sequence_length: int = Field(
        default=1024,
        ge=128,
        le=8192,
    )


    per_device_train_batch_size: int = Field(
        default=1,
        ge=1,
        le=64,
    )


    gradient_accumulation_steps: int = Field(
        default=8,
        ge=1,
        le=1024,
    )


    num_train_epochs: float = Field(
        default=2.0,
        gt=0.0,
        le=20.0,
    )


    learning_rate: float = Field(
        default=2e-4,
        gt=0.0,
        le=1.0,
    )


    warmup_ratio: float = Field(
        default=0.03,
        ge=0.0,
        lt=1.0,
    )


    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


    optimizer: OptimizerName = (
        "paged_adamw_8bit"
    )


    scheduler: SchedulerName = (
        "cosine"
    )


    gradient_checkpointing: Literal[
        True,
    ] = True


    bf16: Literal[
        True,
    ] = True


    fp16: Literal[
        False,
    ] = False


# ============================================================
# DATASET EVIDENCE
# ============================================================


class AdaptationDatasetEvidence(
    BaseModel
):
    """
    Identity and contamination evidence for the frozen adaptation
    dataset.

    Raw training examples are not embedded in this contract.

    The actual dataset must be separately frozen and hashed before
    this contract becomes executable.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    dataset_id: str = Field(
        min_length=1,
    )


    dataset_version: str = Field(
        min_length=1,
    )


    dataset_sha256: str = Field(
        pattern=_SHA256_PATTERN,
    )


    example_count: int = Field(
        gt=0,
    )


    scope: AdaptationDatasetScope = (
        "adaptation_training"
    )


    contamination_report_sha256: str = Field(
        pattern=_SHA256_PATTERN,
    )


    frozen: Literal[
        True,
    ] = True


    contains_regression_expected_answers: Literal[
        False,
    ] = False


    contains_pre_adaptation_holdout_material: Literal[
        False,
    ] = False


    contains_rag_holdout_material: Literal[
        False,
    ] = False


    governance_rule_version: Literal[
        "adaptation_data_governance_v0.1"
    ] = ADAPTATION_DATA_GOVERNANCE_RULE_VERSION


# ============================================================
# EVALUATION EVIDENCE
# ============================================================


class AdaptationEvidenceArtifact(
    BaseModel
):
    """
    Hash-locked evidence referenced by an adaptation experiment.

    The contract records evidence identity only. It does not copy
    benchmark expected answers into the training surface.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    artifact_id: str = Field(
        min_length=1,
    )


    relative_path: str = Field(
        min_length=1,
    )


    sha256: str = Field(
        pattern=_SHA256_PATTERN,
    )


    phase: AdaptationEvidencePhase


    @field_validator(
        "relative_path",
    )
    @classmethod
    def require_relative_path(
        cls,
        value: str,
    ) -> str:
        normalized = (
            value
            .replace(
                "\\",
                "/",
            )
        )


        if (
            normalized.startswith(
                "/"
            )
            or
            ":" in normalized
        ):
            raise ValueError(
                "Evidence artifact path must be repository-relative."
            )


        if (
            ".."
            in normalized.split(
                "/"
            )
        ):
            raise ValueError(
                "Evidence artifact path must not traverse parents."
            )


        return normalized


# ============================================================
# ACCEPTANCE POLICY
# ============================================================


class AdaptationEvaluationPolicy(
    BaseModel
):
    """
    Pre-registered acceptance rules.

    Training loss alone can never accept an adapted model.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    require_regression_gate_pass: Literal[
        True,
    ] = True


    require_safety_gate_pass: Literal[
        True,
    ] = True


    maximum_dangerous_false_positive_increase: Literal[
        0,
    ] = 0


    require_independent_final_acceptance_holdout: Literal[
        True,
    ] = True


    final_acceptance_holdout_must_be_frozen_before_training: Literal[
        True,
    ] = True


    final_acceptance_holdout_must_not_be_used_for_tuning: Literal[
        True,
    ] = True


    training_loss_is_acceptance_evidence: Literal[
        False,
    ] = False


# ============================================================
# QLORA EXPERIMENT CONTRACT
# ============================================================


class QLoRAExperimentContract(
    BaseModel
):
    """
    Immutable, fail-closed contract for one DataLens QLoRA
    adaptation experiment.

    This object is configuration and provenance only.

    It must never contain:
    - Hugging Face access tokens;
    - raw benchmark expected answers;
    - raw holdout cases;
    - arbitrary executable code;
    - model weights;
    - adapter weights;
    - secrets.

    An experiment cannot be considered ready for training unless
    all required dataset and evidence hashes are present.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    experiment_id: str = Field(
        min_length=1,
    )


    adaptation_method: AdaptationMethod = (
        "qlora"
    )


    base_model: BaseModelReference


    quantization: QLoRAQuantizationPolicy = Field(
        default_factory=QLoRAQuantizationPolicy,
    )


    lora: QLoRAParameters = Field(
        default_factory=QLoRAParameters,
    )


    training: AdaptationTrainingPolicy = Field(
        default_factory=AdaptationTrainingPolicy,
    )


    training_dataset: AdaptationDatasetEvidence


    regression_baselines: list[
        AdaptationEvidenceArtifact
    ] = Field(
        min_length=1,
    )


    pre_adaptation_holdouts: list[
        AdaptationEvidenceArtifact
    ] = Field(
        min_length=1,
    )


    final_acceptance_holdout: AdaptationEvidenceArtifact


    evaluation: AdaptationEvaluationPolicy = Field(
        default_factory=AdaptationEvaluationPolicy,
    )


    rule_version: Literal[
        "qlora_experiment_contract_v0.1"
    ] = QLORA_EXPERIMENT_CONTRACT_RULE_VERSION


    @model_validator(
        mode="after",
    )
    def validate_evidence_phases(
        self,
    ) -> "QLoRAExperimentContract":
        invalid_regression = [
            artifact.artifact_id

            for artifact
            in self.regression_baselines

            if (
                artifact.phase
                !=
                "regression_baseline"
            )
        ]


        if invalid_regression:
            raise ValueError(
                "Regression baseline evidence contains artifacts "
                f"with invalid phases: {invalid_regression}."
            )


        invalid_pre_holdout = [
            artifact.artifact_id

            for artifact
            in self.pre_adaptation_holdouts

            if (
                artifact.phase
                !=
                "pre_adaptation_holdout"
            )
        ]


        if invalid_pre_holdout:
            raise ValueError(
                "Pre-adaptation holdout evidence contains artifacts "
                f"with invalid phases: {invalid_pre_holdout}."
            )


        if (
            self.final_acceptance_holdout.phase
            !=
            "final_acceptance_holdout_freeze"
        ):
            raise ValueError(
                "Final acceptance holdout must use phase "
                "final_acceptance_holdout_freeze."
            )


        evidence_hashes = [
            artifact.sha256

            for artifact
            in (
                self.regression_baselines
                +
                self.pre_adaptation_holdouts
                +
                [
                    self.final_acceptance_holdout,
                ]
            )
        ]


        if (
            len(
                evidence_hashes
            )
            !=
            len(
                set(
                    evidence_hashes
                )
            )
        ):
            raise ValueError(
                "Evaluation evidence artifacts must have unique hashes."
            )


        return self