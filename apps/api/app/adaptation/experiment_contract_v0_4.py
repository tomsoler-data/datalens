from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from app.adaptation.contracts import (
    AdaptationDatasetEvidence,
    AdaptationEvaluationPolicy,
    AdaptationEvidenceArtifact,
    AdaptationTrainingPolicy,
    BaseModelReference,
    QLoRAExperimentContract,
    QLoRAParameters,
    QLoRAQuantizationPolicy,
)


QLORA_V04_CONTRACT_FREEZE_RULE_VERSION = (
    "qlora_v0.4_experiment_contract_freeze_v0.1"
)


EXPERIMENT_ID = (
    "datalens-semantic-qlora-v0.4"
)


CONTRACT_RELATIVE_PATH = (
    "artifacts/adaptation/experiments/"
    "datalens_semantic_qlora_v0.4_contract.json"
)


FREEZE_RELATIVE_PATH = (
    "artifacts/adaptation/experiments/"
    "datalens_semantic_qlora_v0.4_contract_freeze.json"
)


# ============================================================
# MODEL
# ============================================================


BASE_MODEL_REPOSITORY = (
    "google/gemma-3-4b-it"
)


BASE_MODEL_REVISION = (
    "093f9f388b31de276ce2de164bdc2081324b9767"
)


# ============================================================
# v0.4 DESIGN AUTHORITY
# ============================================================


DESIGN_RELATIVE_PATH = (
    "artifacts/adaptation/design/"
    "datalens_semantic_qlora_v0.4_"
    "experiment_design_v0.1.json"
)


DESIGN_FREEZE_RELATIVE_PATH = (
    "artifacts/adaptation/design/"
    "datalens_semantic_qlora_v0.4_"
    "experiment_design_v0.1_freeze.json"
)


EXPECTED_DESIGN_SHA256 = (
    "dd36103a01cadc49101dfeffa006bba9"
    "e9cf6cdfc3599a3f0967beff14765cf9"
)


EXPECTED_DESIGN_FREEZE_SHA256 = (
    "84d64a1406ce0e6648de54e4830eb96d"
    "d99b87a980108085906eb57875ef978a"
)


# ============================================================
# OPTIMIZATION AUTHORITY
# ============================================================


OPTIMIZATION_POLICY_RELATIVE_PATH = (
    "artifacts/adaptation/design/"
    "datalens_semantic_qlora_v0.4_"
    "optimization_policy_v0.1.json"
)


OPTIMIZATION_POLICY_FREEZE_RELATIVE_PATH = (
    "artifacts/adaptation/design/"
    "datalens_semantic_qlora_v0.4_"
    "optimization_policy_v0.1_freeze.json"
)


EXPECTED_OPTIMIZATION_POLICY_SHA256 = (
    "01a8fc993a8699e1ae6511f5ce73c642"
    "c7b1c1bf1d974b147ba5e6542d48824d"
)


EXPECTED_OPTIMIZATION_POLICY_FREEZE_SHA256 = (
    "3a5cf4123cefa25afe597ab057e86307"
    "e4b51505f4e7d0d98927f9a42476259c"
)


# ============================================================
# TRAINING DATA
# ============================================================


DATASET_RELATIVE_PATH = (
    "artifacts/adaptation/datasets/"
    "datalens_semantic_training_v0.4.jsonl"
)


DATASET_FREEZE_RELATIVE_PATH = (
    "artifacts/adaptation/datasets/"
    "datalens_semantic_training_v0.4_freeze.json"
)


EXPECTED_DATASET_SHA256 = (
    "4fd00586f2d53d6de57f5cbc5f1d7bfb"
    "2e512960e60b30c28596aaefbac322b7"
)


EXPECTED_DATASET_FREEZE_SHA256 = (
    "c9ae4421becea37ee07bf964f82e2e4f"
    "ffd9328f2920cda7d371fe57e0eb1f70"
)


EXPECTED_CONTAMINATION_REPORT_SHA256 = (
    "2436e108ded2f0b9b3a8dc74b4aecc15"
    "680956418b37cd853066d1bfb18a57ec"
)


EXPECTED_PROVENANCE_REPORT_SHA256 = (
    "1ee13d5e95a6b072f719afcb104741ac"
    "ec8d1255d97500c7e1f425ea4c3d015a"
)


EXPECTED_EXAMPLE_COUNT = 230


# ============================================================
# TOKEN EVIDENCE
# ============================================================


TOKEN_AUDIT_RELATIVE_PATH = (
    "artifacts/adaptation/probes/"
    "datalens_semantic_training_v0.4_"
    "token_length_audit.json"
)


EXPECTED_TOKEN_AUDIT_SHA256 = (
    "add94bd50fc89120a7626fa7533af299"
    "6765d25c7c64d2d866dd7178674d40c1"
)


# ============================================================
# REGRESSION / PRE-ADAPTATION EVIDENCE
# ============================================================


REGRESSION_BASELINES = (
    {
        "artifact_id":
            "semantic-s3-regression-249",

        "relative_path":
            (
                "artifacts/evaluation/experiments/"
                "semantic_s3_regression_249.json"
            ),

        "sha256":
            (
                "fdb5510e9426b857aa9e52feb4d3282f"
                "367e10af1d8ae4335c727673506960ac"
            ),

        "phase":
            "regression_baseline",
    },
)


PRE_ADAPTATION_HOLDOUTS = (
    {
        "artifact_id":
            "semantic-s0-manufacturing-holdout",

        "relative_path":
            (
                "artifacts/evaluation/experiments/"
                "semantic_s0_manufacturing_holdout.json"
            ),

        "sha256":
            (
                "9231548867ceac9832d455a03e01b3db"
                "6617b9ea61b4e83b5e07a4d87f63ef8d"
            ),

        "phase":
            "pre_adaptation_holdout",
    },

    {
        "artifact_id":
            "semantic-s1-logistics-holdout",

        "relative_path":
            (
                "artifacts/evaluation/experiments/"
                "semantic_s1_logistics_holdout.json"
            ),

        "sha256":
            (
                "0520ab2b34f772352bf7de3d1941bb3b"
                "a0ab9f1c432df41c9fef9599fd839e0f"
            ),

        "phase":
            "pre_adaptation_holdout",
    },

    {
        "artifact_id":
            "semantic-s2-cloud-holdout",

        "relative_path":
            (
                "artifacts/evaluation/experiments/"
                "semantic_s2_cloud_holdout.json"
            ),

        "sha256":
            (
                "d27e58c93208f6b28abd5e699ccb65ad"
                "520cc36c49512909aa60a44f9887ce4d"
            ),

        "phase":
            "pre_adaptation_holdout",
    },

    {
        "artifact_id":
            "semantic-s3-customer-support-holdout",

        "relative_path":
            (
                "artifacts/evaluation/experiments/"
                "semantic_s3_customer_support_holdout.json"
            ),

        "sha256":
            (
                "1093f35d6f30870bf128226ecfa32ea7"
                "851aaa1ab5cc2d2a603fccca61735cd2"
            ),

        "phase":
            "pre_adaptation_holdout",
    },

    {
        "artifact_id":
            "semantic-s3-electric-mobility-holdout",

        "relative_path":
            (
                "artifacts/evaluation/experiments/"
                "semantic_s3_electric_mobility_holdout.json"
            ),

        "sha256":
            (
                "6ab32e5e6551555e05d02ed8d7a4e0b"
                "9bbfa980d6ca9984b8d71dd6e12b87402"
            ),

        "phase":
            "pre_adaptation_holdout",
    },
)


# ============================================================
# AIRPORT INDEPENDENT HOLDOUT
# ============================================================


AIRPORT_FREEZE_RELATIVE_PATH = (
    "artifacts/adaptation/holdouts/"
    "datalens_semantic_qlora_v0.4_"
    "airport_ground_operations_holdout_v0.1_freeze.json"
)


EXPECTED_AIRPORT_FREEZE_SHA256 = (
    "46accf23eeae32f0fdc926f7b0a9e731"
    "15a1413ddbc28efa84472f0571ee2f09"
)


EXPECTED_AIRPORT_CASES_SHA256 = (
    "7d9454d0d59dd047050bd72d195feb13"
    "90ca0edc2c83584af0cbfec951cf3939"
)


EXPECTED_AIRPORT_HOLDOUT_ID = (
    "adaptation:datalens-semantic-qlora-v0.4:"
    "airport-ground-operations:holdout:v0.1"
)


# ============================================================
# FINAL ACCEPTANCE
# ============================================================


FINAL_ACCEPTANCE_RELATIVE_PATH = (
    "artifacts/evaluation/holdouts/"
    "greenhouse_operations_final_acceptance_v0.1_freeze.json"
)


EXPECTED_FINAL_ACCEPTANCE_FREEZE_SHA256 = (
    "11615352547f152b91e592066142c343"
    "1b557c5f620018380b71b931cb77a736"
)


# ============================================================
# HISTORICAL PREDECESSOR
# ============================================================


V03_CONTRACT_RELATIVE_PATH = (
    "artifacts/adaptation/experiments/"
    "datalens_semantic_qlora_v0.3_contract.json"
)


V03_CONTRACT_FREEZE_RELATIVE_PATH = (
    "artifacts/adaptation/experiments/"
    "datalens_semantic_qlora_v0.3_contract_freeze.json"
)


EXPECTED_V03_CONTRACT_SHA256 = (
    "609954fe4f06ace47000475053dcb011a"
    "4337e36122678860fb744eff645f92e"
)


EXPECTED_V03_CONTRACT_FREEZE_SHA256 = (
    "9e7eaedefe9823bf070e1c4c25574ee4"
    "2cc0081bd8bf3f1810255609cda4559c"
)


# ============================================================
# HELPERS
# ============================================================


def _root(
    repository_root: Path | None,
) -> Path:
    root = (
        Path.cwd()
        if repository_root is None
        else repository_root
    )

    root = root.expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(
            root
        )

    return root


def _sha256_bytes(
    payload: bytes,
) -> str:
    return hashlib.sha256(
        payload
    ).hexdigest()


def _sha256_file(
    path: Path,
) -> str:
    return _sha256_bytes(
        path.read_bytes()
    )


def _load_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def _require_sha(
    *,
    repository_root: Path,
    relative_path: str,
    expected_sha256: str,
) -> Path:
    path = (
        repository_root
        /
        relative_path
    ).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    actual = _sha256_file(
        path
    )

    if actual != expected_sha256:
        raise RuntimeError(
            (
                "Bound evidence SHA changed.\n"
                f"Path:     {relative_path}\n"
                f"Expected: {expected_sha256}\n"
                f"Actual:   {actual}"
            )
        )

    return path


def _canonical_json_bytes(
    value: Mapping[str, Any],
) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def _git_head(
    *,
    repository_root: Path,
) -> str:
    return subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=repository_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()


def _utc_now() -> str:
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


def _verify_evidence_artifacts(
    *,
    repository_root: Path,
    artifacts: Sequence[Mapping[str, str]],
) -> None:
    for artifact in artifacts:
        _require_sha(
            repository_root=
                repository_root,

            relative_path=
                artifact[
                    "relative_path"
                ],

            expected_sha256=
                artifact[
                    "sha256"
                ],
        )


# ============================================================
# AUTHORITY VALIDATION
# ============================================================


def _validate_authorities(
    *,
    repository_root: Path,
) -> Dict[str, Any]:
    design_path = _require_sha(
        repository_root=
            repository_root,

        relative_path=
            DESIGN_RELATIVE_PATH,

        expected_sha256=
            EXPECTED_DESIGN_SHA256,
    )

    design_freeze_path = _require_sha(
        repository_root=
            repository_root,

        relative_path=
            DESIGN_FREEZE_RELATIVE_PATH,

        expected_sha256=
            EXPECTED_DESIGN_FREEZE_SHA256,
    )

    optimization_path = _require_sha(
        repository_root=
            repository_root,

        relative_path=
            OPTIMIZATION_POLICY_RELATIVE_PATH,

        expected_sha256=
            EXPECTED_OPTIMIZATION_POLICY_SHA256,
    )

    optimization_freeze_path = _require_sha(
        repository_root=
            repository_root,

        relative_path=
            OPTIMIZATION_POLICY_FREEZE_RELATIVE_PATH,

        expected_sha256=
            EXPECTED_OPTIMIZATION_POLICY_FREEZE_SHA256,
    )

    dataset_path = _require_sha(
        repository_root=
            repository_root,

        relative_path=
            DATASET_RELATIVE_PATH,

        expected_sha256=
            EXPECTED_DATASET_SHA256,
    )

    dataset_freeze_path = _require_sha(
        repository_root=
            repository_root,

        relative_path=
            DATASET_FREEZE_RELATIVE_PATH,

        expected_sha256=
            EXPECTED_DATASET_FREEZE_SHA256,
    )

    token_audit_path = _require_sha(
        repository_root=
            repository_root,

        relative_path=
            TOKEN_AUDIT_RELATIVE_PATH,

        expected_sha256=
            EXPECTED_TOKEN_AUDIT_SHA256,
    )

    airport_freeze_path = _require_sha(
        repository_root=
            repository_root,

        relative_path=
            AIRPORT_FREEZE_RELATIVE_PATH,

        expected_sha256=
            EXPECTED_AIRPORT_FREEZE_SHA256,
    )

    final_acceptance_path = _require_sha(
        repository_root=
            repository_root,

        relative_path=
            FINAL_ACCEPTANCE_RELATIVE_PATH,

        expected_sha256=
            EXPECTED_FINAL_ACCEPTANCE_FREEZE_SHA256,
    )

    _require_sha(
        repository_root=
            repository_root,

        relative_path=
            V03_CONTRACT_RELATIVE_PATH,

        expected_sha256=
            EXPECTED_V03_CONTRACT_SHA256,
    )

    _require_sha(
        repository_root=
            repository_root,

        relative_path=
            V03_CONTRACT_FREEZE_RELATIVE_PATH,

        expected_sha256=
            EXPECTED_V03_CONTRACT_FREEZE_SHA256,
    )

    _verify_evidence_artifacts(
        repository_root=
            repository_root,

        artifacts=
            REGRESSION_BASELINES,
    )

    _verify_evidence_artifacts(
        repository_root=
            repository_root,

        artifacts=
            PRE_ADAPTATION_HOLDOUTS,
    )

    design = _load_json(
        design_path
    )

    design_freeze = _load_json(
        design_freeze_path
    )

    optimization = _load_json(
        optimization_path
    )

    optimization_freeze = _load_json(
        optimization_freeze_path
    )

    dataset_freeze = _load_json(
        dataset_freeze_path
    )

    token_audit = _load_json(
        token_audit_path
    )

    airport_freeze = _load_json(
        airport_freeze_path
    )

    final_acceptance_freeze = _load_json(
        final_acceptance_path
    )

    if (
        design[
            "experiment_id"
        ]
        !=
        EXPERIMENT_ID
    ):
        raise RuntimeError(
            "Experiment-design ID mismatch."
        )

    if (
        design_freeze[
            "experiment_id"
        ]
        !=
        EXPERIMENT_ID
    ):
        raise RuntimeError(
            "Experiment-design freeze ID mismatch."
        )

    if (
        optimization[
            "experiment_id"
        ]
        !=
        EXPERIMENT_ID
    ):
        raise RuntimeError(
            "Optimization-policy ID mismatch."
        )

    if (
        optimization_freeze[
            "experiment_id"
        ]
        !=
        EXPERIMENT_ID
    ):
        raise RuntimeError(
            "Optimization freeze ID mismatch."
        )

    if (
        optimization_freeze[
            "policy_sha256"
        ]
        !=
        EXPECTED_OPTIMIZATION_POLICY_SHA256
    ):
        raise RuntimeError(
            "Optimization freeze binding mismatch."
        )

    if (
        optimization_freeze[
            "frozen_before_resource_preflight"
        ]
        is not True
    ):
        raise RuntimeError(
            "Optimization policy was not frozen "
            "before resource preflight."
        )

    if (
        optimization_freeze[
            "frozen_before_v0_4_training"
        ]
        is not True
    ):
        raise RuntimeError(
            "Optimization policy was not frozen "
            "before training."
        )

    if (
        optimization_freeze[
            "training_started_at_freeze"
        ]
        is not False
    ):
        raise RuntimeError(
            "Training had started at optimization freeze."
        )

    dataset = dataset_freeze[
        "dataset"
    ]

    if (
        dataset[
            "dataset_sha256"
        ]
        !=
        EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError(
            "Dataset-freeze binding mismatch."
        )

    if (
        dataset[
            "example_count"
        ]
        !=
        EXPECTED_EXAMPLE_COUNT
    ):
        raise RuntimeError(
            "Dataset example count mismatch."
        )

    if (
        dataset[
            "contamination_report_sha256"
        ]
        !=
        EXPECTED_CONTAMINATION_REPORT_SHA256
    ):
        raise RuntimeError(
            "Dataset contamination binding mismatch."
        )

    if (
        dataset_freeze[
            "provenance_report_sha256"
        ]
        !=
        EXPECTED_PROVENANCE_REPORT_SHA256
    ):
        raise RuntimeError(
            "Dataset provenance binding mismatch."
        )

    if (
        dataset_freeze[
            "contamination_match_count"
        ]
        !=
        0
    ):
        raise RuntimeError(
            "Dataset contamination gate failed."
        )

    if (
        dataset_freeze[
            "provenance_violation_count"
        ]
        !=
        0
    ):
        raise RuntimeError(
            "Dataset provenance gate failed."
        )

    if (
        dataset_freeze[
            "contains_final_acceptance_material"
        ]
        is not False
    ):
        raise RuntimeError(
            "Training dataset contains Final "
            "Acceptance material."
        )

    if (
        dataset_freeze[
            "final_acceptance_tuning_input"
        ]
        is not False
    ):
        raise RuntimeError(
            "Final Acceptance is marked as "
            "training/tuning input."
        )

    if (
        dataset_freeze[
            "frozen_before_training"
        ]
        is not True
    ):
        raise RuntimeError(
            "Training dataset was not frozen "
            "before training."
        )

    if (
        dataset_freeze[
            "training_started_at_freeze"
        ]
        is not False
    ):
        raise RuntimeError(
            "Training had started at dataset freeze."
        )

    if (
        token_audit[
            "dataset"
        ][
            "dataset_sha256"
        ]
        !=
        EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError(
            "Token evidence dataset binding mismatch."
        )

    if (
        token_audit[
            "recommendation"
        ][
            "recommended_max_sequence_length"
        ]
        !=
        256
    ):
        raise RuntimeError(
            "Token evidence no longer supports seq=256."
        )

    if (
        token_audit[
            "recommendation"
        ][
            "truncated_examples"
        ]
        !=
        0
    ):
        raise RuntimeError(
            "Token evidence is no longer lossless."
        )

    if (
        airport_freeze[
            "holdout_id"
        ]
        !=
        EXPECTED_AIRPORT_HOLDOUT_ID
    ):
        raise RuntimeError(
            "Airport holdout ID mismatch."
        )

    if (
        airport_freeze[
            "cases_sha256"
        ]
        !=
        EXPECTED_AIRPORT_CASES_SHA256
    ):
        raise RuntimeError(
            "Airport cases identity mismatch."
        )

    airport_required_true = (
        "independent_holdout",
        "frozen_before_v0_4_training",
        "frozen_before_v0_4_training_data_authoring",
    )

    for key in airport_required_true:
        if airport_freeze[
            key
        ] is not True:
            raise RuntimeError(
                f"Airport holdout gate failed: {key}"
            )

    airport_required_false = (
        "evaluation_executed",
        "results_observed",
        "used_for_hyperparameter_tuning",
        "used_for_training",
        "final_acceptance_opened",
        "v0_4_training_executed",
    )

    for key in airport_required_false:
        if airport_freeze[
            key
        ] is not False:
            raise RuntimeError(
                f"Airport isolation gate failed: {key}"
            )

    if (
        final_acceptance_freeze[
            "adaptation_tuning_input"
        ]
        is not False
    ):
        raise RuntimeError(
            "Final Acceptance is tuning input."
        )

    if (
        final_acceptance_freeze[
            "frozen_before_training"
        ]
        is not True
    ):
        raise RuntimeError(
            "Final Acceptance was not frozen "
            "before training."
        )

    if (
        final_acceptance_freeze[
            "training_started_at_freeze"
        ]
        is not False
    ):
        raise RuntimeError(
            "Training had started at Final "
            "Acceptance freeze."
        )

    # The dataset file is intentionally SHA-checked above,
    # but Airport case content is never opened here.

    return {
        "design":
            design,

        "optimization":
            optimization,

        "dataset":
            dataset,

        "airport_freeze":
            airport_freeze,

        "final_acceptance_freeze":
            final_acceptance_freeze,
    }


# ============================================================
# CONTRACT
# ============================================================


def build_experiment_contract(
    *,
    repository_root: Path,
) -> QLoRAExperimentContract:
    repository_root = _root(
        repository_root
    )

    authorities = _validate_authorities(
        repository_root=
            repository_root,
    )

    optimization = authorities[
        "optimization"
    ]

    training_authority = optimization[
        "training"
    ]

    contract = QLoRAExperimentContract(
        experiment_id=
            EXPERIMENT_ID,

        base_model=
            BaseModelReference(
                repository=
                    BASE_MODEL_REPOSITORY,

                revision=
                    BASE_MODEL_REVISION,

                tokenizer_revision=
                    BASE_MODEL_REVISION,

                model_family=
                    "gemma3",

                modality_scope=
                    "text_only",

                trust_remote_code=
                    False,

                use_multimodal_inputs=
                    False,
            ),

        quantization=
            QLoRAQuantizationPolicy(
                load_in_4bit=
                    True,

                quantization_type=
                    "nf4",

                use_double_quantization=
                    True,

                compute_dtype=
                    "bfloat16",
            ),

        lora=
            QLoRAParameters(
                rank=
                    16,

                alpha=
                    32,

                dropout=
                    0.05,

                bias=
                    "none",

                task_type=
                    "CAUSAL_LM",

                target_strategy=
                    "language_model_all_linear",

                target_resolver_rule_version=
                    "qlora_target_resolver_v0.1",
            ),

        training=
            AdaptationTrainingPolicy(
                random_seed=
                    training_authority[
                        "random_seed"
                    ],

                max_sequence_length=
                    training_authority[
                        "max_sequence_length"
                    ],

                per_device_train_batch_size=
                    training_authority[
                        "micro_batch_size"
                    ],

                gradient_accumulation_steps=
                    training_authority[
                        "gradient_accumulation_steps"
                    ],

                num_train_epochs=
                    float(
                        training_authority[
                            "initial_epoch_budget"
                        ]
                    ),

                learning_rate=
                    training_authority[
                        "learning_rate"
                    ],

                warmup_ratio=
                    training_authority[
                        "warmup_ratio"
                    ],

                weight_decay=
                    training_authority[
                        "weight_decay"
                    ],

                optimizer=
                    training_authority[
                        "optimizer"
                    ],

                scheduler=
                    training_authority[
                        "scheduler"
                    ],

                gradient_checkpointing=
                    training_authority[
                        "gradient_checkpointing"
                    ],

                bf16=
                    training_authority[
                        "bf16"
                    ],

                fp16=
                    training_authority[
                        "fp16"
                    ],
            ),

        training_dataset=
            AdaptationDatasetEvidence(
                dataset_id=
                    (
                        "adaptation:datalens-semantic:"
                        "training:v0.4"
                    ),

                dataset_version=
                    (
                        "datalens_semantic_"
                        "adaptation_training_v0.4"
                    ),

                dataset_sha256=
                    EXPECTED_DATASET_SHA256,

                example_count=
                    EXPECTED_EXAMPLE_COUNT,

                scope=
                    "adaptation_training",

                contamination_report_sha256=
                    EXPECTED_CONTAMINATION_REPORT_SHA256,

                frozen=
                    True,

                contains_regression_expected_answers=
                    False,

                contains_pre_adaptation_holdout_material=
                    False,

                contains_rag_holdout_material=
                    False,

                governance_rule_version=
                    "adaptation_data_governance_v0.1",
            ),

        regression_baselines=[
            AdaptationEvidenceArtifact(
                **artifact
            )

            for artifact
            in REGRESSION_BASELINES
        ],

        pre_adaptation_holdouts=[
            AdaptationEvidenceArtifact(
                **artifact
            )

            for artifact
            in PRE_ADAPTATION_HOLDOUTS
        ],

        final_acceptance_holdout=
            AdaptationEvidenceArtifact(
                artifact_id=
                    (
                        "final-acceptance:"
                        "greenhouse-operations:v0.1"
                    ),

                relative_path=
                    FINAL_ACCEPTANCE_RELATIVE_PATH,

                sha256=
                    EXPECTED_FINAL_ACCEPTANCE_FREEZE_SHA256,

                phase=
                    "final_acceptance_holdout_freeze",
            ),

        evaluation=
            AdaptationEvaluationPolicy(
                require_regression_gate_pass=
                    True,

                require_safety_gate_pass=
                    True,

                maximum_dangerous_false_positive_increase=
                    0,

                require_independent_final_acceptance_holdout=
                    True,

                final_acceptance_holdout_must_be_frozen_before_training=
                    True,

                final_acceptance_holdout_must_not_be_used_for_tuning=
                    True,

                training_loss_is_acceptance_evidence=
                    False,
            ),
    )

    return contract


def contract_dict(
    *,
    repository_root: Path,
) -> Dict[str, Any]:
    contract = build_experiment_contract(
        repository_root=
            repository_root
    )

    payload = contract.model_dump(
        mode="json"
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "Contract serialization is not an object."
        )

    return payload


def contract_bytes(
    *,
    repository_root: Path,
) -> bytes:
    return _canonical_json_bytes(
        contract_dict(
            repository_root=
                repository_root
        )
    )


def contract_sha256(
    *,
    repository_root: Path,
) -> str:
    return _sha256_bytes(
        contract_bytes(
            repository_root=
                repository_root
        )
    )


# ============================================================
# VALIDATION
# ============================================================


def validate_contract(
    *,
    repository_root: Path,
) -> Dict[str, Any]:
    repository_root = _root(
        repository_root
    )

    first = contract_dict(
        repository_root=
            repository_root
    )

    second = contract_dict(
        repository_root=
            repository_root
    )

    first_bytes = _canonical_json_bytes(
        first
    )

    second_bytes = _canonical_json_bytes(
        second
    )

    if first_bytes != second_bytes:
        raise RuntimeError(
            "Experiment contract is not deterministic."
        )

    validated = QLoRAExperimentContract.model_validate(
        first
    )

    if (
        validated.experiment_id
        !=
        EXPERIMENT_ID
    ):
        raise RuntimeError(
            "Experiment contract ID changed."
        )

    if (
        validated.training_dataset.dataset_sha256
        !=
        EXPECTED_DATASET_SHA256
    ):
        raise RuntimeError(
            "Contract dataset binding changed."
        )

    if (
        validated.training_dataset.example_count
        !=
        EXPECTED_EXAMPLE_COUNT
    ):
        raise RuntimeError(
            "Contract dataset count changed."
        )

    expected_training = {
        "random_seed":
            42,

        "max_sequence_length":
            256,

        "per_device_train_batch_size":
            1,

        "gradient_accumulation_steps":
            8,

        "num_train_epochs":
            2.0,

        "learning_rate":
            0.0002,

        "warmup_ratio":
            0.03,

        "weight_decay":
            0.0,

        "optimizer":
            "paged_adamw_8bit",

        "scheduler":
            "cosine",

        "gradient_checkpointing":
            True,

        "bf16":
            True,

        "fp16":
            False,
    }

    training_payload = first[
        "training"
    ]

    for key, expected in expected_training.items():
        if (
            training_payload[
                key
            ]
            !=
            expected
        ):
            raise RuntimeError(
                (
                    "Contract training value changed: "
                    f"{key}"
                )
            )

    if (
        len(
            first[
                "regression_baselines"
            ]
        )
        !=
        1
    ):
        raise RuntimeError(
            "Regression baseline count changed."
        )

    if (
        len(
            first[
                "pre_adaptation_holdouts"
            ]
        )
        !=
        5
    ):
        raise RuntimeError(
            "Pre-adaptation holdout count changed."
        )

    pre_paths = {
        item[
            "relative_path"
        ]

        for item
        in first[
            "pre_adaptation_holdouts"
        ]
    }

    if any(
        "airport"
        in
        path.casefold()

        for path
        in pre_paths
    ):
        raise RuntimeError(
            "Airport was inserted into "
            "pre-adaptation evidence."
        )

    if (
        first[
            "final_acceptance_holdout"
        ][
            "sha256"
        ]
        !=
        EXPECTED_FINAL_ACCEPTANCE_FREEZE_SHA256
    ):
        raise RuntimeError(
            "Final Acceptance binding changed."
        )

    if (
        first[
            "evaluation"
        ][
            "training_loss_is_acceptance_evidence"
        ]
        is not False
    ):
        raise RuntimeError(
            "Training loss became acceptance evidence."
        )

    return first


# ============================================================
# FREEZE
# ============================================================


def build_contract_freeze(
    *,
    repository_root: Path,
    contract_sha256_value: str,
) -> Dict[str, Any]:
    repository_root = _root(
        repository_root
    )

    authorities = _validate_authorities(
        repository_root=
            repository_root
    )

    airport_freeze = authorities[
        "airport_freeze"
    ]

    return {
        "freeze_id":
            (
                "experiment-contract-freeze:"
                "datalens-semantic-qlora:v0.4"
            ),

        "rule_version":
            QLORA_V04_CONTRACT_FREEZE_RULE_VERSION,

        "experiment_contract_rule_version":
            "qlora_experiment_contract_v0.1",

        "experiment_id":
            EXPERIMENT_ID,

        "status":
            "frozen",

        "contract_relative_path":
            CONTRACT_RELATIVE_PATH,

        "contract_sha256":
            contract_sha256_value,

        "source_git_commit":
            _git_head(
                repository_root=
                    repository_root
            ),

        "supersedes": {
            "experiment_id":
                "datalens-semantic-qlora-v0.3",

            "contract_relative_path":
                V03_CONTRACT_RELATIVE_PATH,

            "contract_sha256":
                EXPECTED_V03_CONTRACT_SHA256,

            "contract_freeze_relative_path":
                V03_CONTRACT_FREEZE_RELATIVE_PATH,

            "contract_freeze_sha256":
                EXPECTED_V03_CONTRACT_FREEZE_SHA256,
        },

        "authorities": {
            "experiment_design": {
                "relative_path":
                    DESIGN_RELATIVE_PATH,

                "sha256":
                    EXPECTED_DESIGN_SHA256,

                "freeze_relative_path":
                    DESIGN_FREEZE_RELATIVE_PATH,

                "freeze_sha256":
                    EXPECTED_DESIGN_FREEZE_SHA256,
            },

            "optimization_policy": {
                "relative_path":
                    OPTIMIZATION_POLICY_RELATIVE_PATH,

                "sha256":
                    EXPECTED_OPTIMIZATION_POLICY_SHA256,

                "freeze_relative_path":
                    OPTIMIZATION_POLICY_FREEZE_RELATIVE_PATH,

                "freeze_sha256":
                    EXPECTED_OPTIMIZATION_POLICY_FREEZE_SHA256,
            },

            "token_length_evidence": {
                "relative_path":
                    TOKEN_AUDIT_RELATIVE_PATH,

                "sha256":
                    EXPECTED_TOKEN_AUDIT_SHA256,
            },
        },

        "training_dataset": {
            "relative_path":
                DATASET_RELATIVE_PATH,

            "dataset_sha256":
                EXPECTED_DATASET_SHA256,

            "freeze_relative_path":
                DATASET_FREEZE_RELATIVE_PATH,

            "freeze_sha256":
                EXPECTED_DATASET_FREEZE_SHA256,

            "example_count":
                EXPECTED_EXAMPLE_COUNT,

            "contamination_report_sha256":
                EXPECTED_CONTAMINATION_REPORT_SHA256,

            "provenance_report_sha256":
                EXPECTED_PROVENANCE_REPORT_SHA256,

            "contamination_match_count":
                0,

            "provenance_violation_count":
                0,
        },

        "optimization_execution_plan": {
            "micro_batches_per_epoch":
                230,

            "full_accumulation_groups_per_epoch":
                28,

            "partial_accumulation_group_size":
                6,

            "optimizer_steps_per_epoch":
                29,

            "total_micro_batches":
                460,

            "total_optimizer_steps":
                58,

            "example_presentations":
                460,

            "discarded_example_presentations":
                0,

            "cross_epoch_accumulation":
                False,

            "partial_group_policy":
                "flush_partial_group_at_epoch_end",
        },

        "regression_baseline_sha256": [
            artifact[
                "sha256"
            ]

            for artifact
            in REGRESSION_BASELINES
        ],

        "pre_adaptation_holdout_sha256": [
            artifact[
                "sha256"
            ]

            for artifact
            in PRE_ADAPTATION_HOLDOUTS
        ],

        "airport_independent_holdout": {
            "holdout_id":
                EXPECTED_AIRPORT_HOLDOUT_ID,

            "freeze_relative_path":
                AIRPORT_FREEZE_RELATIVE_PATH,

            "freeze_sha256":
                EXPECTED_AIRPORT_FREEZE_SHA256,

            "cases_sha256":
                EXPECTED_AIRPORT_CASES_SHA256,

            "independent_holdout":
                True,

            "used_for_training":
                False,

            "used_for_hyperparameter_tuning":
                False,

            "evaluation_executed_at_contract_freeze":
                False,

            "results_observed_at_contract_freeze":
                False,
        },

        "final_acceptance": {
            "freeze_relative_path":
                FINAL_ACCEPTANCE_RELATIVE_PATH,

            "freeze_sha256":
                EXPECTED_FINAL_ACCEPTANCE_FREEZE_SHA256,

            "tuning_input":
                False,

            "cases_loaded_at_contract_freeze":
                False,

            "evaluated_at_contract_freeze":
                False,
        },

        "frozen_before_resource_preflight":
            True,

        "frozen_before_training":
            True,

        "training_started_at_contract_freeze":
            False,

        "model_loaded_at_contract_freeze":
            False,

        "optimizer_created_at_contract_freeze":
            False,

        "airport_evaluated_at_contract_freeze":
            False,

        "final_acceptance_loaded_at_contract_freeze":
            False,

        "final_acceptance_evaluated_at_contract_freeze":
            False,

        "frozen_at":
            _utc_now(),
    }


def _publish_new_bundle(
    *,
    outputs: Mapping[
        Path,
        bytes,
    ],
) -> None:
    for path in outputs:
        if path.exists():
            raise FileExistsError(
                path
            )

    temporary: Dict[
        Path,
        Path,
    ] = {}

    published = []

    try:
        for path, payload in outputs.items():
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temp = (
                path.parent
                /
                (
                    "."
                    +
                    path.name
                    +
                    "."
                    +
                    uuid.uuid4().hex
                    +
                    ".tmp"
                )
            )

            with temp.open(
                "xb"
            ) as handle:
                handle.write(
                    payload
                )

            if (
                _sha256_file(
                    temp
                )
                !=
                _sha256_bytes(
                    payload
                )
            ):
                raise RuntimeError(
                    "Temporary artifact SHA mismatch."
                )

            temporary[
                path
            ] = temp

        for path, temp in temporary.items():
            os.replace(
                temp,
                path,
            )

            published.append(
                path
            )

    except Exception:
        for path in reversed(
            published
        ):
            if path.exists():
                path.unlink()

        raise

    finally:
        for temp in temporary.values():
            if temp.exists():
                temp.unlink()


def freeze_contract(
    *,
    repository_root: Path,
) -> Dict[str, Any]:
    repository_root = _root(
        repository_root
    )

    contract = validate_contract(
        repository_root=
            repository_root
    )

    contract_payload = _canonical_json_bytes(
        contract
    )

    contract_sha = _sha256_bytes(
        contract_payload
    )

    freeze = build_contract_freeze(
        repository_root=
            repository_root,

        contract_sha256_value=
            contract_sha,
    )

    freeze_payload = _canonical_json_bytes(
        freeze
    )

    contract_path = (
        repository_root
        /
        CONTRACT_RELATIVE_PATH
    ).resolve()

    freeze_path = (
        repository_root
        /
        FREEZE_RELATIVE_PATH
    ).resolve()

    _publish_new_bundle(
        outputs={
            contract_path:
                contract_payload,

            freeze_path:
                freeze_payload,
        }
    )

    return freeze


# ============================================================
# VERIFICATION
# ============================================================


def verify_contract_artifacts(
    *,
    repository_root: Path,
) -> Dict[str, str]:
    repository_root = _root(
        repository_root
    )

    contract_path = (
        repository_root
        /
        CONTRACT_RELATIVE_PATH
    ).resolve()

    freeze_path = (
        repository_root
        /
        FREEZE_RELATIVE_PATH
    ).resolve()

    if not contract_path.is_file():
        raise FileNotFoundError(
            contract_path
        )

    if not freeze_path.is_file():
        raise FileNotFoundError(
            freeze_path
        )

    expected_contract_bytes = (
        contract_bytes(
            repository_root=
                repository_root
        )
    )

    actual_contract_bytes = (
        contract_path.read_bytes()
    )

    if (
        actual_contract_bytes
        !=
        expected_contract_bytes
    ):
        raise RuntimeError(
            "Published contract differs from "
            "deterministic recomputation."
        )

    freeze = _load_json(
        freeze_path
    )

    actual_contract_sha = _sha256_bytes(
        actual_contract_bytes
    )

    if (
        freeze[
            "contract_sha256"
        ]
        !=
        actual_contract_sha
    ):
        raise RuntimeError(
            "Contract freeze SHA binding mismatch."
        )

    if (
        freeze[
            "rule_version"
        ]
        !=
        QLORA_V04_CONTRACT_FREEZE_RULE_VERSION
    ):
        raise RuntimeError(
            "Contract freeze rule mismatch."
        )

    return {
        "contract_sha256":
            actual_contract_sha,

        "freeze_sha256":
            _sha256_file(
                freeze_path
            ),
    }


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=(
            "validate",
            "freeze",
            "verify",
        ),
    )

    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
    )

    args = parser.parse_args()

    root = _root(
        args.repository_root
    )

    if args.command == "validate":
        contract_path = (
            root
            /
            CONTRACT_RELATIVE_PATH
        )

        freeze_path = (
            root
            /
            FREEZE_RELATIVE_PATH
        )

        if (
            contract_path.exists()
            or
            freeze_path.exists()
        ):
            raise RuntimeError(
                "Official v0.4 contract artifacts "
                "already exist."
            )

        contract = validate_contract(
            repository_root=
                root
        )

        print(
            "=== DATALENS QLORA v0.4 EXPERIMENT CONTRACT v0.1 ==="
        )

        print()

        print(
            (
                "Experiment ID: "
                f"{contract['experiment_id']}"
            )
        )

        print(
            (
                "Rule version: "
                f"{contract['rule_version']}"
            )
        )

        print(
            (
                "Future contract SHA256: "
                f"{contract_sha256(repository_root=root)}"
            )
        )

        print()

        print(
            "TRAINING DATA"
        )

        print(
            (
                "  Dataset: "
                f"{contract['training_dataset']['dataset_id']}"
            )
        )

        print(
            (
                "  Examples: "
                f"{contract['training_dataset']['example_count']}"
            )
        )

        print()

        print(
            "TRAINING"
        )

        for key, value in contract[
            "training"
        ].items():
            print(
                f"  {key}: {value}"
            )

        print()

        print(
            "EVALUATION BOUNDARY"
        )

        print(
            (
                "  Regression baselines: "
                f"{len(contract['regression_baselines'])}"
            )
        )

        print(
            (
                "  Pre-adaptation holdouts: "
                f"{len(contract['pre_adaptation_holdouts'])}"
            )
        )

        print(
            "  Airport in pre-adaptation list: False"
        )

        print(
            "  Final Acceptance frozen reference: True"
        )

        print()

        print(
            "SAFETY"
        )

        print(
            "  Official artifacts written: False"
        )

        print(
            "  Airport cases opened: False"
        )

        print(
            "  Airport evaluated: False"
        )

        print(
            "  Final Acceptance cases opened: False"
        )

        print(
            "  Model loaded: False"
        )

        print(
            "  CUDA requested: False"
        )

        print(
            "  Training executed: False"
        )

        print()

        print(
            "DATALENS QLORA v0.4 EXPERIMENT CONTRACT v0.1: PASS"
        )

        return

    if args.command == "freeze":
        freeze_contract(
            repository_root=
                root
        )

        print(
            "DATALENS QLORA v0.4 EXPERIMENT CONTRACT FREEZE: PASS"
        )

        return

    result = verify_contract_artifacts(
        repository_root=
            root
    )

    print(
        "=== DATALENS QLORA v0.4 EXPERIMENT CONTRACT VERIFY v0.1 ==="
    )

    print()

    print(
        (
            "Contract SHA256: "
            f"{result['contract_sha256']}"
        )
    )

    print(
        (
            "Freeze SHA256: "
            f"{result['freeze_sha256']}"
        )
    )

    print(
        "Deterministic contract identity: PASS"
    )

    print()

    print(
        "DATALENS QLORA v0.4 EXPERIMENT CONTRACT VERIFY v0.1: PASS"
    )


if __name__ == "__main__":
    main()
