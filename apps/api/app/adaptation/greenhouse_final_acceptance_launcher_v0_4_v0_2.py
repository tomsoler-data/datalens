from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from app.adaptation import (
    greenhouse_final_acceptance_runner_v0_4_v0_4 as runner,
)


GREENHOUSE_FINAL_ACCEPTANCE_LAUNCHER_RULE_VERSION = (
    "qlora_v0.4_greenhouse_final_acceptance_launcher_v0.2"
)

EXPERIMENT_ID = (
    "adaptation:datalens-semantic-qlora-v0.4"
)

FINAL_ACCEPTANCE_REVISION = "v0.1"

CASE_COUNT = 18

RANDOM_SEED = 42


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

REPOSITORY_ROOT = (
    ROOT
    .parents[1]
)


LAUNCHER_PATH = Path(__file__).resolve()


# ============================================================
# Frozen non-protected authorities
# ============================================================

EXPECTED_RUNNER_SHA256 = (
    "8e922f46d65048ab6bcebeab8eca7e52b8bf83100524bea84658530039b40cea"
)

EXPECTED_LAUNCH_PATH_DECISION_SHA256 = (
    "ccf9d9127eca9690dc4b8742356bf9673a79bbae9d23e8b0d54b948b776f00c4"
)

EXPECTED_PROTOCOL_SHA256 = (
    "056272cca075a6f8c48c38f5296c0ddacaaeb29e64f08048a43a80ea3dab0b16"
)

EXPECTED_TASK_INTERPRETATION_SHA256 = (
    "9f097f8b8e7e2dd182cc86cc5ed12fc74ded18262bac2af3639d2bf07bdf61a2"
)

EXPECTED_SERIALIZER_DECISION_SHA256 = (
    "df54af3e32dc97b5fd43573c8c2a8caf1335976fde3b39faa51afc48ab5244e3"
)

EXPECTED_ADAPTER_BUNDLE_SHA256 = (
    "0351980df6d86096195c0971deb30c725e155c71aa5de8054b2b37fa42090716"
)

EXPECTED_ADAPTER_FILES = {
    "README.md":
        (
            "6ecdbb662eaed8010ab0e012a2b95b79543884cf294406dc6da2cde64f98389d"
        ),

    "adapter_config.json":
        (
            "3ae14896612f6bf74ee7786a450e2ac0f08f3da9f33391505cb1a7dc823dcdb8"
        ),

    "adapter_model.safetensors":
        (
            "4f145b0bf37f67841c09f02b86679634a9532491d2f560b0e7c5c328009e4610"
        ),
}


RUNNER_PATH = (
    ROOT
    / "app"
    / "adaptation"
    / "greenhouse_final_acceptance_runner_v0_4_v0_4.py"
)

LAUNCH_PATH_DECISION_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
    / "datalens_semantic_qlora_v0.4_"
      "greenhouse_execution_launch_path_decision_v0.1.json"
)

PROTOCOL_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
    / "datalens_semantic_qlora_v0.4_"
      "greenhouse_final_acceptance_protocol_v0.1.json"
)

TASK_INTERPRETATION_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
    / "datalens_semantic_qlora_v0.4_"
      "greenhouse_gate_task_interpretation_decision_v0.1.json"
)

SERIALIZER_DECISION_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
    / "datalens_semantic_qlora_v0.4_"
      "greenhouse_deterministic_input_serializer_decision_v0.1.json"
)

PREPARATION_AUTHORIZATION_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
    / "datalens_semantic_qlora_v0.4_"
      "greenhouse_final_acceptance_authorization_v0.1.json"
)

ADAPTER_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "adapters"
    / "datalens_semantic_qlora_v0.4_adapter"
)


# ============================================================
# IMPORTANT
#
# This v0.2 authorization does NOT exist yet.
#
# The launcher therefore cannot execute the official Greenhouse
# until the launcher itself, its revised Manifest, its synthetic
# launch-path preflight, and the rebound authorization have all
# been frozen and committed.
# ============================================================

DEFAULT_EXECUTION_AUTHORIZATION_PATH = (
    ROOT
    / "artifacts"
    / "adaptation"
    / "evaluation"
    / "datalens_semantic_qlora_v0.4_"
      "greenhouse_final_acceptance_execution_authorization_v0.3.json"
)


# ============================================================
# Exact v0.1 single-use output names remain unchanged.
# ============================================================

EXPECTED_CONSUMPTION_OUTPUT = (
    "artifacts/adaptation/evaluation/"
    "datalens_semantic_qlora_v0.4_"
    "greenhouse_final_acceptance_v0.1_consumption.json"
)

EXPECTED_REPORT_OUTPUT = (
    "artifacts/adaptation/evaluation/"
    "datalens_semantic_qlora_v0.4_"
    "greenhouse_final_acceptance_v0.1_report.json"
)

EXPECTED_RECEIPT_OUTPUT = (
    "artifacts/adaptation/evaluation/"
    "datalens_semantic_qlora_v0.4_"
    "greenhouse_final_acceptance_v0.1_receipt.json"
)


EXPECTED_RELATIONS = (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
)


POSITIVE_RELATIONS = {
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
}

NEGATIVE_OR_NONCOMMITTAL_RELATIONS = {
    "unrelated",
    "uncertain",
}


# ============================================================
# Generic helpers
# ============================================================

def sha256_bytes(
    payload: bytes,
) -> str:

    return hashlib.sha256(
        payload
    ).hexdigest()


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        for chunk in iter(
            lambda:
                handle.read(
                    1024 * 1024
                ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


def canonical_json_bytes(
    payload: Any,
) -> bytes:

    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def git_bytes(
    *args: str,
) -> bytes:

    return subprocess.run(
        [
            "git",
            *args,
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout


def git_text(
    *args: str,
) -> str:

    return (
        git_bytes(
            *args
        )
        .decode(
            "utf-8"
        )
        .strip()
    )


def git_head() -> str:

    return git_text(
        "rev-parse",
        "HEAD",
    )


def git_parent() -> str:

    return git_text(
        "rev-parse",
        "HEAD^",
    )


def api_relative_to_repo_path(
    api_relative_path: str,
) -> str:

    normalized = (
        api_relative_path
        .replace(
            "\\",
            "/",
        )
        .lstrip(
            "/"
        )
    )

    return (
        "apps/api/"
        +
        normalized
    )


def require_clean_repository() -> None:

    status = git_text(
        "status",
        "--porcelain",
    )

    if status:

        raise RuntimeError(
            (
                "Repository must be clean before "
                "Greenhouse consumption.\n"
                f"{status}"
            )
        )


def require_exact_sha(
    *,
    path: Path,
    expected_sha256: str,
    label: str,
) -> None:

    actual = sha256_file(
        path
    )

    if actual != expected_sha256:

        raise RuntimeError(
            (
                f"{label} SHA mismatch.\n"
                f"Expected: {expected_sha256}\n"
                f"Actual:   {actual}"
            )
        )


def load_json(
    path: Path,
) -> Any:

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def resolve_api_output_path(
    relative_path: str,
) -> Path:

    normalized = (
        relative_path
        .replace(
            "\\",
            "/",
        )
        .lstrip(
            "/"
        )
    )

    candidate = (
        ROOT
        /
        normalized
    ).resolve()

    root = ROOT.resolve()

    if not candidate.is_relative_to(
        root
    ):

        raise RuntimeError(
            (
                "Output path escaped apps/api root: "
                f"{relative_path}"
            )
        )

    return candidate


# ============================================================
# Atomic output primitives
# ============================================================

def atomic_write_json_exclusive(
    *,
    path: Path,
    payload: Mapping[
        str,
        Any,
    ],
) -> None:

    path = path.resolve()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    flags = (
        os.O_WRONLY
        |
        os.O_CREAT
        |
        os.O_EXCL
    )

    try:

        fd = os.open(
            str(
                path
            ),
            flags,
            0o600,
        )

    except FileExistsError as exc:

        raise RuntimeError(
            (
                "Exclusive single-use artifact already exists: "
                f"{path}"
            )
        ) from exc


    try:

        payload_bytes = (
            canonical_json_bytes(
                payload
            )
        )

        with os.fdopen(
            fd,
            "wb",
        ) as handle:

            handle.write(
                payload_bytes
            )

            handle.flush()

            os.fsync(
                handle.fileno()
            )

    except Exception:

        raise


def create_consumption_marker(
    *,
    path: Path,
    payload: Mapping[
        str,
        Any,
    ],
) -> None:

    atomic_write_json_exclusive(
        path=
            path,
        payload=
            payload,
    )


# ============================================================
# Protected-material gate
# ============================================================

class ProtectedMaterialGate:

    def __init__(
        self,
        *,
        marker_path: Path,
    ) -> None:

        self._marker_path = (
            marker_path
            .resolve()
        )


    def require_consumed(
        self,
    ) -> None:

        if not self._marker_path.is_file():

            raise RuntimeError(
                (
                    "Protected Greenhouse material cannot be read "
                    "before the single-use consumption marker exists."
                )
            )


    def read_verified_bytes(
        self,
        *,
        path: Path,
        expected_sha256: str,
        expected_size_bytes: int,
        label: str,
    ) -> bytes:

        self.require_consumed()

        payload = (
            path
            .resolve()
            .read_bytes()
        )

        actual_size = len(
            payload
        )

        if actual_size != expected_size_bytes:

            raise RuntimeError(
                (
                    f"{label} size mismatch after consumption.\n"
                    f"Expected: {expected_size_bytes}\n"
                    f"Actual:   {actual_size}"
                )
            )


        actual_sha256 = sha256_bytes(
            payload
        )

        if actual_sha256 != expected_sha256:

            raise RuntimeError(
                (
                    f"{label} SHA mismatch after consumption.\n"
                    f"Expected: {expected_sha256}\n"
                    f"Actual:   {actual_sha256}"
                )
            )


        return payload


# ============================================================
# Frozen gold relation projection
# ============================================================

def project_gold_relation(
    *,
    same_concept: bool,
    same_concept_family: bool,
    same_domain: bool,
    distinct_variants: bool,
    compatible_units: bool,
    derived_gap_compatible: bool,
) -> str:

    _ = compatible_units
    _ = derived_gap_compatible


    if (
        same_concept
        and
        not same_concept_family
    ):

        raise RuntimeError(
            (
                "Gold projection consistency violation: "
                "same_concept=True with "
                "same_concept_family=False."
            )
        )


    if (
        same_concept
        and
        not same_domain
    ):

        raise RuntimeError(
            (
                "Gold projection consistency violation: "
                "same_concept=True with same_domain=False."
            )
        )


    if (
        same_concept_family
        and
        not same_domain
    ):

        raise RuntimeError(
            (
                "Gold projection consistency violation: "
                "same_concept_family=True with "
                "same_domain=False."
            )
        )


    if (
        same_concept
        and
        distinct_variants
    ):

        return (
            "same_metric_different_state"
        )


    if (
        not same_concept
        and
        same_concept_family
    ):

        return (
            "related_distinct_metric"
        )


    if (
        not same_concept
        and
        not same_concept_family
        and
        not same_domain
    ):

        return (
            "unrelated"
        )


    return (
        "uncertain"
    )


# ============================================================
# Protected cases parser
# ============================================================

def _is_case_identity_dict(
    value: Any,
) -> bool:

    if not isinstance(
        value,
        dict,
    ):

        return False


    for key in (
        "case_id",
        "left_column",
        "right_column",
    ):

        if not isinstance(
            value.get(
                key
            ),
            str,
        ):

            return False


        if not value[
            key
        ]:

            return False


    return True


def _find_case_lists(
    value: Any,
    output: list[
        list[
            dict[
                str,
                Any,
            ]
        ]
    ],
) -> None:

    if isinstance(
        value,
        list,
    ):

        if (
            len(
                value
            )
            ==
            CASE_COUNT
            and
            all(
                _is_case_identity_dict(
                    item
                )
                for item
                in value
            )
        ):

            output.append(
                value
            )


        for item in value:

            _find_case_lists(
                item,
                output,
            )


    elif isinstance(
        value,
        dict,
    ):

        for child in value.values():

            _find_case_lists(
                child,
                output,
            )


def extract_protected_case_entries(
    payload: Any,
) -> list[
    dict[
        str,
        Any,
    ]
]:

    candidates: list[
        list[
            dict[
                str,
                Any,
            ]
        ]
    ] = []


    _find_case_lists(
        payload,
        candidates,
    )


    unique: dict[
        str,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = {}


    for candidate in candidates:

        fingerprint = (
            json.dumps(
                candidate,
                sort_keys=True,
                ensure_ascii=True,
                separators=(
                    ",",
                    ":",
                ),
            )
        )

        unique[
            fingerprint
        ] = candidate


    if len(
        unique
    ) != 1:

        raise RuntimeError(
            (
                "Expected exactly one unique protected "
                "18-case identity list; found "
                f"{len(unique)}."
            )
        )


    cases = next(
        iter(
            unique.values()
        )
    )


    case_ids = [
        case[
            "case_id"
        ]
        for case
        in cases
    ]


    if len(
        set(
            case_ids
        )
    ) != CASE_COUNT:

        raise RuntimeError(
            "Protected case IDs are not unique."
        )


    return [
        dict(
            case
        )
        for case
        in cases
    ]


# ============================================================
# Post-consumption benchmark validation
# ============================================================

def load_frozen_greenhouse_benchmark_after_consumption(
    *,
    protected_gate: ProtectedMaterialGate,
) -> Any:

    protected_gate.require_consumed()

    return importlib.import_module(
        "app.evaluation.benchmarks."
        "greenhouse_operations_final_acceptance"
    )


def build_gold_case_records(
    *,
    benchmark_module: Any,
    protected_case_entries: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:

    benchmark_cases = (
        benchmark_module
        .build_greenhouse_final_acceptance_pair_cases()
    )


    if len(
        benchmark_cases
    ) != CASE_COUNT:

        raise RuntimeError(
            (
                "Frozen benchmark pair-case count changed. "
                f"Expected={CASE_COUNT}, "
                f"actual={len(benchmark_cases)}"
            )
        )


    protected_by_id = {
        str(
            case[
                "case_id"
            ]
        ):
            case

        for case
        in protected_case_entries
    }


    output: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for benchmark_case in benchmark_cases:

        case_id = str(
            benchmark_case.case_id
        )


        protected_case = (
            protected_by_id.get(
                case_id
            )
        )


        if protected_case is None:

            raise RuntimeError(
                (
                    "Protected case manifest is missing "
                    f"benchmark case {case_id!r}."
                )
            )


        left_column = str(
            benchmark_case.left_column
        )

        right_column = str(
            benchmark_case.right_column
        )


        if (
            protected_case[
                "left_column"
            ]
            !=
            left_column
        ):

            raise RuntimeError(
                (
                    "Protected/benchmark left-column mismatch "
                    f"for {case_id!r}."
                )
            )


        if (
            protected_case[
                "right_column"
            ]
            !=
            right_column
        ):

            raise RuntimeError(
                (
                    "Protected/benchmark right-column mismatch "
                    f"for {case_id!r}."
                )
            )


        dataset_id = str(
            benchmark_case.left_dataset_id
        )


        if (
            str(
                benchmark_case.right_dataset_id
            )
            !=
            dataset_id
        ):

            raise RuntimeError(
                (
                    "Greenhouse benchmark case uses "
                    "different dataset identities."
                )
            )


        for optional_dataset_field in (
            "left_dataset_id",
            "right_dataset_id",
        ):

            if optional_dataset_field in protected_case:

                if (
                    protected_case[
                        optional_dataset_field
                    ]
                    !=
                    dataset_id
                ):

                    raise RuntimeError(
                        (
                            "Protected case dataset identity "
                            f"mismatch: {case_id!r}"
                        )
                    )


        assertions = {
            "same_concept":
                bool(
                    benchmark_case.same_concept
                ),

            "same_concept_family":
                bool(
                    benchmark_case.same_concept_family
                ),

            "same_domain":
                bool(
                    benchmark_case.same_domain
                ),

            "distinct_variants":
                bool(
                    benchmark_case.distinct_variants
                ),

            "compatible_units":
                bool(
                    benchmark_case.compatible_units
                ),

            "derived_gap_compatible":
                bool(
                    benchmark_case.derived_gap_compatible
                ),
        }


        for key, expected_value in assertions.items():

            if key in protected_case:

                if (
                    protected_case[
                        key
                    ]
                    is not expected_value
                ):

                    raise RuntimeError(
                        (
                            "Protected/benchmark assertion mismatch "
                            f"for {case_id!r}: {key}"
                        )
                    )


        expected_relation = (
            project_gold_relation(
                **assertions
            )
        )


        output.append(
            {
                "case_id":
                    case_id,

                "left_dataset_id":
                    dataset_id,

                "right_dataset_id":
                    dataset_id,

                "left_column":
                    left_column,

                "right_column":
                    right_column,

                "assertions":
                    assertions,

                "expected_relation":
                    expected_relation,
            }
        )


    if len(
        output
    ) != CASE_COUNT:

        raise RuntimeError(
            "Gold case projection count changed."
        )


    return output


# ============================================================
# Dataset validation
# ============================================================

def parse_and_validate_dataset(
    *,
    dataset_bytes: bytes,
    benchmark_module: Any,
) -> pd.DataFrame:

    dataframe = pd.read_csv(
        io.BytesIO(
            dataset_bytes
        )
    )


    expected = (
        benchmark_module
        .build_greenhouse_final_acceptance_dataframe()
    )


    if list(
        dataframe.columns
    ) != list(
        expected.columns
    ):

        raise RuntimeError(
            "Protected Greenhouse dataset columns changed."
        )


    if dataframe.shape != expected.shape:

        raise RuntimeError(
            (
                "Protected Greenhouse dataframe shape changed. "
                f"Expected={expected.shape}, "
                f"actual={dataframe.shape}"
            )
        )


    try:

        pd.testing.assert_frame_equal(
            dataframe,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )

    except AssertionError as exc:

        raise RuntimeError(
            "Protected Greenhouse dataframe content mismatch."
        ) from exc


    return dataframe


# ============================================================
# Independence supporting material
# ============================================================

def validate_independence_material(
    payload: Any,
) -> None:

    if not isinstance(
        payload,
        (
            dict,
            list,
        ),
    ):

        raise RuntimeError(
            (
                "Greenhouse independence supporting "
                "material must be JSON object/list."
            )
        )


    if isinstance(
        payload,
        dict,
    ) and not payload:

        raise RuntimeError(
            "Greenhouse independence JSON object is empty."
        )


    if isinstance(
        payload,
        list,
    ) and not payload:

        raise RuntimeError(
            "Greenhouse independence JSON list is empty."
        )


# ============================================================
# Evaluation helpers
# ============================================================

def build_prebuilt_prompt_records(
    *,
    cases: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    profile_index: Mapping[
        tuple[
            str,
            str,
        ],
        Any,
    ],
) -> tuple[
    dict[
        str,
        Any,
    ],
    ...,
]:

    if len(
        cases
    ) != CASE_COUNT:

        raise RuntimeError(
            "Prompt prebuild must receive exactly 18 cases."
        )


    records: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for case in cases:

        case_identity = {
            "left_dataset_id":
                case[
                    "left_dataset_id"
                ],

            "right_dataset_id":
                case[
                    "right_dataset_id"
                ],

            "left_column":
                case[
                    "left_column"
                ],

            "right_column":
                case[
                    "right_column"
                ],
        }


        user_message = (
            runner.build_label_blind_user_message(
                case_identity=
                    case_identity,
                profile_index=
                    profile_index,
            )
        )


        if (
            not isinstance(
                user_message,
                str,
            )
            or
            not user_message
        ):

            raise RuntimeError(
                "Prebuilt label-blind message is invalid."
            )


        records.append(
            {
                "case_identity":
                    case_identity,

                "user_message":
                    user_message,
            }
        )


    if len(
        records
    ) != CASE_COUNT:

        raise RuntimeError(
            "Prompt prebuild count changed."
        )


    return tuple(
        records
    )


def validate_prebuilt_prompt_records(
    *,
    cases: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    prebuilt_prompt_records: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:

    if len(
        cases
    ) != CASE_COUNT:

        raise RuntimeError(
            "Prompt validation must receive exactly 18 cases."
        )


    if len(
        prebuilt_prompt_records
    ) != CASE_COUNT:

        raise RuntimeError(
            "Expected exactly 18 prebuilt prompt records."
        )


    for index, case in enumerate(
        cases
    ):

        record = (
            prebuilt_prompt_records[
                index
            ]
        )


        if set(
            record
        ) != {
            "case_identity",
            "user_message",
        }:

            raise RuntimeError(
                "Prebuilt prompt-record key set changed."
            )


        expected_identity = {
            "left_dataset_id":
                case[
                    "left_dataset_id"
                ],

            "right_dataset_id":
                case[
                    "right_dataset_id"
                ],

            "left_column":
                case[
                    "left_column"
                ],

            "right_column":
                case[
                    "right_column"
                ],
        }


        if dict(
            record[
                "case_identity"
            ]
        ) != expected_identity:

            raise RuntimeError(
                (
                    "Prebuilt prompt identity/order mismatch "
                    f"at index {index}."
                )
            )


        user_message = (
            record[
                "user_message"
            ]
        )


        if (
            not isinstance(
                user_message,
                str,
            )
            or
            not user_message
        ):

            raise RuntimeError(
                (
                    "Prebuilt user message invalid "
                    f"at index {index}."
                )
            )


def evaluate_model_once(
    *,
    model: Any,
    tokenizer: Any,
    cases: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    prebuilt_prompt_records: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    torch_module: Any,
) -> dict[
    str,
    Any,
]:
    if len(
        cases
    ) != CASE_COUNT:

        raise RuntimeError(
            "Evaluation must receive exactly 18 cases."
        )


    records: list[
        dict[
            str,
            Any,
        ]
    ] = []


    correct_count = 0
    strict_json_valid_count = 0
    dangerous_false_positives = 0
    safety_failures = 0


    by_relation: dict[
        str,
        dict[
            str,
            int,
        ],
    ] = {}




    if len(
        prebuilt_prompt_records
    ) != CASE_COUNT:

        raise RuntimeError(
            "Evaluation must receive exactly 18 prebuilt prompts."
        )


    for index, case in enumerate(
        cases
    ):

        prompt_record = (
            prebuilt_prompt_records[
                index
            ]
        )


        generated = (
            runner.generate_label_blind_case_from_user_message(
                model=
                    model,
                tokenizer=
                    tokenizer,
                user_message=
                    prompt_record[
                        "user_message"
                    ],
                torch_module=
                    torch_module,
            )
        )


        predicted_relation = (
            generated.get(
                "relation"
            )
        )

        strict_json_valid = bool(
            generated.get(
                "strict_json_valid",
                False,
            )
        )

        expected_relation = str(
            case[
                "expected_relation"
            ]
        )


        correct = (
            strict_json_valid
            and
            predicted_relation
            ==
            expected_relation
        )


        if correct:

            correct_count += 1


        if strict_json_valid:

            strict_json_valid_count += 1


        dangerous_fp = (
            predicted_relation
            ==
            "same_metric_different_state"
            and
            expected_relation
            !=
            "same_metric_different_state"
        )


        if dangerous_fp:

            dangerous_false_positives += 1


        safety_failure = (
            predicted_relation
            in
            POSITIVE_RELATIONS
            and
            expected_relation
            in
            NEGATIVE_OR_NONCOMMITTAL_RELATIONS
        )


        if safety_failure:

            safety_failures += 1


        relation_bucket = (
            by_relation
            .setdefault(
                expected_relation,
                {
                    "count":
                        0,
                    "correct":
                        0,
                },
            )
        )


        relation_bucket[
            "count"
        ] += 1


        if correct:

            relation_bucket[
                "correct"
            ] += 1


        records.append(
            {
                "case_id":
                    case[
                        "case_id"
                    ],

                "left_column":
                    case[
                        "left_column"
                    ],

                "right_column":
                    case[
                        "right_column"
                    ],

                "expected_relation":
                    expected_relation,

                "predicted_relation":
                    predicted_relation,

                "correct":
                    correct,

                "dangerous_false_positive":
                    dangerous_fp,

                "safety_failure":
                    safety_failure,

                "generation":
                    generated,
            }
        )


    accuracy = (
        correct_count
        /
        CASE_COUNT
    )


    strict_json_validity_rate = (
        strict_json_valid_count
        /
        CASE_COUNT
    )


    per_relation_accuracy: dict[
        str,
        float,
    ] = {}


    for relation, bucket in by_relation.items():

        per_relation_accuracy[
            relation
        ] = (
            bucket[
                "correct"
            ]
            /
            bucket[
                "count"
            ]
        )


    macro_accuracy = (
        sum(
            per_relation_accuracy.values()
        )
        /
        len(
            per_relation_accuracy
        )
        if per_relation_accuracy
        else 0.0
    )


    return {
        "case_count":
            CASE_COUNT,

        "correct_count":
            correct_count,

        "accuracy":
            round(
                accuracy,
                6,
            ),

        "macro_accuracy":
            round(
                macro_accuracy,
                6,
            ),

        "strict_json_valid_count":
            strict_json_valid_count,

        "strict_json_validity_rate":
            round(
                strict_json_validity_rate,
                6,
            ),

        "dangerous_false_positives":
            dangerous_false_positives,

        "safety_failures":
            safety_failures,

        "per_relation_accuracy":
            {
                key:
                    round(
                        value,
                        6,
                    )

                for key, value
                in sorted(
                    per_relation_accuracy.items()
                )
            },

        "cases":
            records,
    }


def paired_comparison(
    *,
    base: Mapping[
        str,
        Any,
    ],
    adapted: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:

    return {
        "accuracy_delta":
            round(
                float(
                    adapted[
                        "accuracy"
                    ]
                )
                -
                float(
                    base[
                        "accuracy"
                    ]
                ),
                6,
            ),

        "macro_accuracy_delta":
            round(
                float(
                    adapted[
                        "macro_accuracy"
                    ]
                )
                -
                float(
                    base[
                        "macro_accuracy"
                    ]
                ),
                6,
            ),
    }


def evaluate_acceptance_gates(
    *,
    adapted: Mapping[
        str,
        Any,
    ],
    freeze_integrity_pass: bool,
    s3_regression_pass: bool,
) -> dict[
    str,
    Any,
]:

    gates = {
        "projected_relation_accuracy": {
            "actual":
                adapted[
                    "accuracy"
                ],

            "minimum":
                0.8,

            "pass":
                (
                    float(
                        adapted[
                            "accuracy"
                        ]
                    )
                    >=
                    0.8
                ),
        },

        "strict_json_validity": {
            "actual":
                adapted[
                    "strict_json_validity_rate"
                ],

            "required":
                1.0,

            "pass":
                (
                    float(
                        adapted[
                            "strict_json_validity_rate"
                        ]
                    )
                    ==
                    1.0
                ),
        },

        "dangerous_false_positives": {
            "actual":
                adapted[
                    "dangerous_false_positives"
                ],

            "maximum":
                0,

            "pass":
                (
                    int(
                        adapted[
                            "dangerous_false_positives"
                        ]
                    )
                    ==
                    0
                ),
        },

        "safety_failures": {
            "actual":
                adapted[
                    "safety_failures"
                ],

            "maximum":
                0,

            "pass":
                (
                    int(
                        adapted[
                            "safety_failures"
                        ]
                    )
                    ==
                    0
                ),
        },

        "freeze_integrity": {
            "actual":
                freeze_integrity_pass,

            "required":
                True,

            "pass":
                freeze_integrity_pass
                is True,
        },

        "s3_regression": {
            "actual":
                (
                    "PASS"
                    if s3_regression_pass
                    else
                    "FAIL"
                ),

            "required":
                "PASS",

            "pass":
                s3_regression_pass
                is True,
        },
    }


    all_passed = all(
        gate[
            "pass"
        ]
        for gate
        in gates.values()
    )


    return {
        "all_passed":
            all_passed,

        "gates":
            gates,

        "failure_action":
            (
                "accept_candidate"
                if all_passed
                else
                "terminal_final_acceptance_failure_no_reexecution"
            ),
    }


# ============================================================
# Rebound Execution Authorization
# ============================================================

def validate_rebound_execution_authorization(
    *,
    authorization_path: Path,
) -> tuple[
    dict[
        str,
        Any,
    ],
    str,
]:

    if not authorization_path.is_file():

        raise RuntimeError(
            (
                "Rebound Greenhouse Execution Authorization "
                "v0.3 is missing. Official execution remains blocked."
            )
        )


    authorization_bytes = (
        authorization_path
        .read_bytes()
    )


    authorization_sha256 = (
        sha256_bytes(
            authorization_bytes
        )
    )


    authorization = json.loads(
        authorization_bytes.decode(
            "utf-8"
        )
    )


    if (
        authorization.get(
            "authorization_rule_version"
        )
        !=
        "qlora_v0.4_greenhouse_final_acceptance_execution_authorization_v0.3"
    ):

        raise RuntimeError(
            "Expected rebound Execution Authorization v0.3."
        )


    if (
        authorization.get(
            "final_acceptance_revision"
        )
        !=
        FINAL_ACCEPTANCE_REVISION
    ):

        raise RuntimeError(
            "Final Acceptance revision changed."
        )


    scope = (
        authorization.get(
            "authorization_scope"
        )
    )


    if not isinstance(
        scope,
        dict,
    ):

        raise RuntimeError(
            "Authorization scope missing."
        )


    for key in (
        "execute_greenhouse",
        "read_greenhouse_dataset",
        "read_greenhouse_cases",
        "read_greenhouse_independence_material",
        "observe_greenhouse_labels",
        "observe_greenhouse_results",
    ):

        if scope.get(
            key
        ) is not True:

            raise RuntimeError(
                (
                    "Required execution authority missing: "
                    f"{key}"
                )
            )


    for key in (
        "reexecute_greenhouse",
        "tune_on_greenhouse",
        "use_greenhouse_for_model_selection",
        "modify_candidate_after_greenhouse",
        "modify_acceptance_gates",
        "modify_protocol",
        "modify_prompt_policy",
        "modify_serializer",
        "modify_generation_policy",
    ):

        if scope.get(
            key
        ) is not False:

            raise RuntimeError(
                (
                    "Forbidden execution authority enabled: "
                    f"{key}"
                )
            )


    activation = (
        authorization.get(
            "activation"
        )
    )


    if not isinstance(
        activation,
        dict,
    ):

        raise RuntimeError(
            "Authorization activation contract missing."
        )


    if (
        activation.get(
            "effective_only_after_git_commit"
        )
        is not True
    ):

        raise RuntimeError(
            "Authorization is not commit-gated."
        )


    if (
        activation.get(
            "authorization_must_be_committed_before_execution"
        )
        is not True
    ):

        raise RuntimeError(
            "Authorization commit requirement changed."
        )


    if (
        activation.get(
            "execution_must_use_committed_authorization_bytes"
        )
        is not True
    ):

        raise RuntimeError(
            "Committed authorization byte requirement changed."
        )


    required_parent = (
        activation.get(
            "required_parent_git_commit"
        )
    )


    if (
        required_parent
        !=
        git_parent()
    ):

        raise RuntimeError(
            (
                "Rebound authorization parent binding mismatch.\n"
                f"Expected current HEAD parent: {git_parent()}\n"
                f"Authorization: {required_parent}"
            )
        )


    relative_path = (
        authorization_path
        .resolve()
        .relative_to(
            ROOT.resolve()
        )
        .as_posix()
    )


    repo_path = (
        api_relative_to_repo_path(
            relative_path
        )
    )


    committed_bytes = git_bytes(
        "show",
        f"HEAD:{repo_path}",
    )


    if committed_bytes != authorization_bytes:

        raise RuntimeError(
            (
                "Execution must use exactly the authorization "
                "bytes committed at HEAD."
            )
        )


    return (
        authorization,
        authorization_sha256,
    )


# ============================================================
# Non-protected frozen authorities
# ============================================================

def validate_non_protected_frozen_authorities(
    *,
    authorization: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:

    require_exact_sha(
        path=
            RUNNER_PATH,
        expected_sha256=
            EXPECTED_RUNNER_SHA256,
        label=
            "Greenhouse Runner v0.4",
    )

    require_exact_sha(
        path=
            LAUNCH_PATH_DECISION_PATH,
        expected_sha256=
            EXPECTED_LAUNCH_PATH_DECISION_SHA256,
        label=
            "Greenhouse Launch-path Decision v0.1",
    )

    require_exact_sha(
        path=
            PROTOCOL_PATH,
        expected_sha256=
            EXPECTED_PROTOCOL_SHA256,
        label=
            "Greenhouse Protocol v0.1",
    )

    require_exact_sha(
        path=
            TASK_INTERPRETATION_PATH,
        expected_sha256=
            EXPECTED_TASK_INTERPRETATION_SHA256,
        label=
            "Greenhouse Task Interpretation v0.1",
    )

    require_exact_sha(
        path=
            SERIALIZER_DECISION_PATH,
        expected_sha256=
            EXPECTED_SERIALIZER_DECISION_SHA256,
        label=
            "Greenhouse Serializer Decision v0.1",
    )


    for filename, expected_sha in (
        EXPECTED_ADAPTER_FILES.items()
    ):

        require_exact_sha(
            path=
                ADAPTER_PATH
                /
                filename,
            expected_sha256=
                expected_sha,
            label=
                (
                    "Frozen adapter file "
                    f"{filename}"
                ),
        )


    if (
        runner.EXPECTED_ADAPTER_BUNDLE_SHA256
        !=
        EXPECTED_ADAPTER_BUNDLE_SHA256
    ):

        raise RuntimeError(
            "Runner adapter-bundle authority changed."
        )


    execution_authorities = (
        authorization.get(
            "execution_authorities"
        )
    )


    if not isinstance(
        execution_authorities,
        dict,
    ):

        raise RuntimeError(
            "Rebound execution-authority bindings missing."
        )


    launcher_binding = (
        execution_authorities.get(
            "launcher"
        )
    )


    if not isinstance(
        launcher_binding,
        dict,
    ):

        raise RuntimeError(
            "Rebound authorization does not bind launcher."
        )


    launcher_sha = (
        sha256_file(
            LAUNCHER_PATH
        )
    )


    if (
        launcher_binding.get(
            "sha256"
        )
        !=
        launcher_sha
    ):

        raise RuntimeError(
            "Rebound authorization launcher SHA mismatch."
        )


    if (
        execution_authorities.get(
            "launch_path_decision",
            {},
        ).get(
            "sha256"
        )
        !=
        EXPECTED_LAUNCH_PATH_DECISION_SHA256
    ):

        raise RuntimeError(
            "Rebound authorization launch-decision binding changed."
        )


    manifest_binding = (
        execution_authorities.get(
            "execution_manifest"
        )
    )


    if not isinstance(
        manifest_binding,
        dict,
    ):

        raise RuntimeError(
            "Rebound authorization Manifest binding missing."
        )


    manifest_relative_path = (
        manifest_binding.get(
            "relative_path"
        )
    )


    manifest_sha256 = (
        manifest_binding.get(
            "sha256"
        )
    )


    if not isinstance(
        manifest_relative_path,
        str,
    ):

        raise RuntimeError(
            "Manifest relative path missing."
        )


    if not isinstance(
        manifest_sha256,
        str,
    ):

        raise RuntimeError(
            "Manifest SHA missing."
        )


    manifest_path = (
        ROOT
        /
        manifest_relative_path
    ).resolve()


    require_exact_sha(
        path=
            manifest_path,
        expected_sha256=
            manifest_sha256,
        label=
            "Rebound Greenhouse execution Manifest",
    )


    manifest = load_json(
        manifest_path
    )


    preparation_binding = (
        authorization.get(
            "preparation_authority"
        )
    )


    if not isinstance(
        preparation_binding,
        dict,
    ):

        raise RuntimeError(
            "Preparation authority binding missing."
        )


    preparation_sha = (
        preparation_binding.get(
            "sha256"
        )
    )


    if not isinstance(
        preparation_sha,
        str,
    ):

        raise RuntimeError(
            "Preparation authority SHA missing."
        )


    require_exact_sha(
        path=
            PREPARATION_AUTHORIZATION_PATH,
        expected_sha256=
            preparation_sha,
        label=
            "Greenhouse preparation authorization",
    )


    preparation = load_json(
        PREPARATION_AUTHORIZATION_PATH
    )


    if (
        preparation[
            "release_conditions"
        ][
            "s3_gate_passed"
        ]
        is not True
    ):

        raise RuntimeError(
            "Frozen S3 regression evidence is not PASS."
        )


    authority, tokenizer = (
        runner.prepare_runtime_authority()
    )


    return {
        "manifest":
            manifest,

        "manifest_path":
            manifest_path,

        "manifest_sha256":
            manifest_sha256,

        "runtime_authority":
            authority,

        "tokenizer":
            tokenizer,

        "s3_regression_pass":
            True,

        "launcher_sha256":
            launcher_sha,
    }


# ============================================================
# Manifest contracts
# ============================================================

def validate_manifest_output_contract(
    manifest: Mapping[
        str,
        Any,
    ],
) -> tuple[
    Path,
    Path,
    Path,
]:

    outputs = (
        manifest.get(
            "outputs"
        )
    )


    if not isinstance(
        outputs,
        dict,
    ):

        raise RuntimeError(
            "Manifest output contract missing."
        )


    if (
        outputs.get(
            "consumption_marker"
        )
        !=
        EXPECTED_CONSUMPTION_OUTPUT
    ):

        raise RuntimeError(
            "Consumption output revision/path changed."
        )


    if (
        outputs.get(
            "report"
        )
        !=
        EXPECTED_REPORT_OUTPUT
    ):

        raise RuntimeError(
            "Report output revision/path changed."
        )


    if (
        outputs.get(
            "receipt"
        )
        !=
        EXPECTED_RECEIPT_OUTPUT
    ):

        raise RuntimeError(
            "Receipt output revision/path changed."
        )


    return (
        resolve_api_output_path(
            EXPECTED_CONSUMPTION_OUTPUT
        ),
        resolve_api_output_path(
            EXPECTED_REPORT_OUTPUT
        ),
        resolve_api_output_path(
            EXPECTED_RECEIPT_OUTPUT
        ),
    )


def require_single_use_outputs_absent(
    *,
    marker_path: Path,
    report_path: Path,
    receipt_path: Path,
) -> None:

    for label, path in (
        (
            "consumption marker",
            marker_path,
        ),
        (
            "report",
            report_path,
        ),
        (
            "receipt",
            receipt_path,
        ),
    ):

        if path.exists():

            raise RuntimeError(
                (
                    "Greenhouse single-use execution blocked: "
                    f"{label} already exists at {path}"
                )
            )


def protected_component_specs(
    manifest: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:

    holdout = (
        manifest.get(
            "protected_holdout"
        )
    )


    if not isinstance(
        holdout,
        dict,
    ):

        raise RuntimeError(
            "Manifest protected_holdout missing."
        )


    if (
        holdout.get(
            "case_count"
        )
        !=
        CASE_COUNT
    ):

        raise RuntimeError(
            "Manifest Greenhouse case count changed."
        )


    components = (
        holdout.get(
            "components"
        )
    )


    if not isinstance(
        components,
        list,
    ):

        raise RuntimeError(
            "Manifest protected component list missing."
        )


    expected_roles = {
        "dataset_definition",
        "case_manifest",
        "supporting_definition",
    }


    by_role: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}


    for component in components:

        if not isinstance(
            component,
            dict,
        ):

            raise RuntimeError(
                "Invalid protected component record."
            )


        role = component.get(
            "role"
        )


        if role in by_role:

            raise RuntimeError(
                f"Duplicate protected component role: {role}"
            )


        if role not in expected_roles:

            raise RuntimeError(
                f"Unexpected protected component role: {role}"
            )


        if (
            component.get(
                "protected_before_consumption"
            )
            is not True
        ):

            raise RuntimeError(
                "Protected component lost sealed status."
            )


        repo_path = component.get(
            "repo_path"
        )

        expected_sha = component.get(
            "sha256"
        )

        expected_size = component.get(
            "size_bytes"
        )


        if not isinstance(
            repo_path,
            str,
        ):

            raise RuntimeError(
                "Protected component repo path missing."
            )


        if not isinstance(
            expected_sha,
            str,
        ):

            raise RuntimeError(
                "Protected component SHA missing."
            )


        if not isinstance(
            expected_size,
            int,
        ):

            raise RuntimeError(
                "Protected component size missing."
            )


        subprocess.run(
            [
                "git",
                "ls-files",
                "--error-unmatch",
                repo_path,
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        )


        by_role[
            role
        ] = dict(
            component
        )


    if set(
        by_role
    ) != expected_roles:

        raise RuntimeError(
            "Protected component-role set changed."
        )


    return by_role


# ============================================================
# Official execution
# ============================================================

def execute_greenhouse_final_acceptance(
    *,
    authorization_path: Optional[
        Path
    ] = None,
) -> dict[
    str,
    Any,
]:

    # --------------------------------------------------------
    # PRE-CONSUMPTION PHASE
    # --------------------------------------------------------

    if authorization_path is None:

        authorization_path = (
            DEFAULT_EXECUTION_AUTHORIZATION_PATH
        )


    (
        authorization,
        authorization_sha256,
    ) = (
        validate_rebound_execution_authorization(
            authorization_path=
                authorization_path,
        )
    )


    safe = (
        validate_non_protected_frozen_authorities(
            authorization=
                authorization,
        )
    )


    require_clean_repository()


    manifest = safe[
        "manifest"
    ]


    (
        marker_path,
        report_path,
        receipt_path,
    ) = (
        validate_manifest_output_contract(
            manifest
        )
    )


    require_single_use_outputs_absent(
        marker_path=
            marker_path,
        report_path=
            report_path,
        receipt_path=
            receipt_path,
    )


    components = (
        protected_component_specs(
            manifest
        )
    )


    # --------------------------------------------------------
    # IRREVERSIBLE SINGLE-USE BOUNDARY
    # --------------------------------------------------------

    marker = {
        "marker_rule_version":
            "qlora_v0.4_greenhouse_final_acceptance_consumption_v0.1",

        "experiment_id":
            EXPERIMENT_ID,

        "holdout_id":
            (
                "adaptation:"
                "datalens-semantic-qlora-v0.4:"
                "greenhouse-operations:"
                "final-acceptance:v0.1"
            ),

        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_head":
            git_head(),

        "execution_authorization_sha256":
            authorization_sha256,

        "execution_manifest_sha256":
            safe[
                "manifest_sha256"
            ],

        "launcher_sha256":
            safe[
                "launcher_sha256"
            ],

        "runner_sha256":
            EXPECTED_RUNNER_SHA256,

        "launch_path_decision_sha256":
            EXPECTED_LAUNCH_PATH_DECISION_SHA256,

        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA256,

        "task_interpretation_sha256":
            EXPECTED_TASK_INTERPRETATION_SHA256,

        "serializer_decision_sha256":
            EXPECTED_SERIALIZER_DECISION_SHA256,

        "protected_bytes_read_before_marker":
            False,

        "protected_hashes_recomputed_before_marker":
            False,

        "protected_json_parsed_before_marker":
            False,

        "greenhouse_results_observed_before_marker":
            False,

        "single_use_consumption_started":
            True,

        "failure_after_marker_is_terminal":
            True,

        "retry_permitted":
            False,

        "reexecution_permitted":
            False,

        "tuning_permitted":
            False,

        "candidate_replacement_permitted":
            False,

        "status":
            "consumption_started",
    }


    create_consumption_marker(
        path=
            marker_path,
        payload=
            marker,
    )


    # --------------------------------------------------------
    # FROM HERE ONWARD, ANY FAILURE IS TERMINAL.
    # --------------------------------------------------------

    protected_gate = (
        ProtectedMaterialGate(
            marker_path=
                marker_path,
        )
    )


    protected_gate.require_consumed()


    dataset_spec = (
        components[
            "dataset_definition"
        ]
    )

    case_spec = (
        components[
            "case_manifest"
        ]
    )

    independence_spec = (
        components[
            "supporting_definition"
        ]
    )


    dataset_bytes = (
        protected_gate
        .read_verified_bytes(
            path=
                REPOSITORY_ROOT
                /
                dataset_spec[
                    "repo_path"
                ],
            expected_sha256=
                dataset_spec[
                    "sha256"
                ],
            expected_size_bytes=
                dataset_spec[
                    "size_bytes"
                ],
            label=
                "Protected Greenhouse dataset",
        )
    )


    cases_bytes = (
        protected_gate
        .read_verified_bytes(
            path=
                REPOSITORY_ROOT
                /
                case_spec[
                    "repo_path"
                ],
            expected_sha256=
                case_spec[
                    "sha256"
                ],
            expected_size_bytes=
                case_spec[
                    "size_bytes"
                ],
            label=
                "Protected Greenhouse case manifest",
        )
    )


    independence_bytes = (
        protected_gate
        .read_verified_bytes(
            path=
                REPOSITORY_ROOT
                /
                independence_spec[
                    "repo_path"
                ],
            expected_sha256=
                independence_spec[
                    "sha256"
                ],
            expected_size_bytes=
                independence_spec[
                    "size_bytes"
                ],
            label=
                "Protected Greenhouse independence material",
        )
    )


    cases_payload = json.loads(
        cases_bytes.decode(
            "utf-8-sig"
        )
    )


    independence_payload = (
        json.loads(
            independence_bytes.decode(
                "utf-8-sig"
            )
        )
    )


    protected_case_entries = (
        extract_protected_case_entries(
            cases_payload
        )
    )


    validate_independence_material(
        independence_payload
    )


    benchmark_module = (
        load_frozen_greenhouse_benchmark_after_consumption(
            protected_gate=
                protected_gate,
        )
    )


    gold_cases = (
        build_gold_case_records(
            benchmark_module=
                benchmark_module,
            protected_case_entries=
                protected_case_entries,
        )
    )


    dataframe = (
        parse_and_validate_dataset(
            dataset_bytes=
                dataset_bytes,
            benchmark_module=
                benchmark_module,
        )
    )


    dataset_id = (
        benchmark_module
        .GREENHOUSE_FINAL_ACCEPTANCE_DATASET_ID
    )

    filename = (
        benchmark_module
        .GREENHOUSE_FINAL_ACCEPTANCE_FILENAME
    )


    semantic_profile = (
        runner
        .build_greenhouse_deterministic_profile(
            dataset_id=
                dataset_id,
            filename=
                filename,
            dataframe=
                dataframe,
        )
    )


    profile_index = (
        runner
        .build_profile_index(
            [
                semantic_profile
            ]
        )
    )


    prebuilt_prompt_records = (
        build_prebuilt_prompt_records(
            cases=
                gold_cases,
            profile_index=
                profile_index,
        )
    )


    validate_prebuilt_prompt_records(
        cases=
            gold_cases,
        prebuilt_prompt_records=
            prebuilt_prompt_records,
    )


    import torch


    torch.manual_seed(
        RANDOM_SEED
    )

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


    authority = safe[
        "runtime_authority"
    ]

    tokenizer = safe[
        "tokenizer"
    ]


    model = (
        runner
        .load_base_model(
            torch_module=
                torch,
            authority=
                authority,
        )
    )


    base_result = (
        evaluate_model_once(
            model=
                model,
            tokenizer=
                tokenizer,
            cases=
                gold_cases,
            prebuilt_prompt_records=
                prebuilt_prompt_records,
            torch_module=
                torch,
        )
    )


    adapted_model = (
        runner
        .attach_adapter(
            model=
                model,
        )
    )


    adapted_result = (
        evaluate_model_once(
            model=
                adapted_model,
            tokenizer=
                tokenizer,
            cases=
                gold_cases,
            prebuilt_prompt_records=
                prebuilt_prompt_records,
            torch_module=
                torch,
        )
    )


    paired = (
        paired_comparison(
            base=
                base_result,
            adapted=
                adapted_result,
        )
    )


    freeze_integrity_pass = True


    gates = (
        evaluate_acceptance_gates(
            adapted=
                adapted_result,
            freeze_integrity_pass=
                freeze_integrity_pass,
            s3_regression_pass=
                bool(
                    safe[
                        "s3_regression_pass"
                    ]
                ),
        )
    )


    created_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )


    report = {
        "report_rule_version":
            "qlora_v0.4_greenhouse_final_acceptance_report_v0.1",

        "launcher_rule_version":
            GREENHOUSE_FINAL_ACCEPTANCE_LAUNCHER_RULE_VERSION,

        "experiment_id":
            EXPERIMENT_ID,

        "final_acceptance_revision":
            FINAL_ACCEPTANCE_REVISION,

        "holdout_id":
            marker[
                "holdout_id"
            ],

        "created_at":
            created_at,

        "git_head":
            git_head(),

        "status":
            "completed",

        "single_use":
            True,

        "execution_authorization_sha256":
            authorization_sha256,

        "execution_manifest_sha256":
            safe[
                "manifest_sha256"
            ],

        "launcher_sha256":
            safe[
                "launcher_sha256"
            ],

        "runner_sha256":
            EXPECTED_RUNNER_SHA256,

        "consumption_marker_sha256":
            sha256_file(
                marker_path
            ),

        "protected_material": {
            "dataset_sha256":
                dataset_spec[
                    "sha256"
                ],

            "cases_sha256":
                case_spec[
                    "sha256"
                ],

            "independence_sha256":
                independence_spec[
                    "sha256"
                ],

            "dataset_size_bytes":
                dataset_spec[
                    "size_bytes"
                ],

            "cases_size_bytes":
                case_spec[
                    "size_bytes"
                ],

            "independence_size_bytes":
                independence_spec[
                    "size_bytes"
                ],

            "freeze_integrity":
                "PASS",
        },

        "generation": {
            "mode":
                "deterministic_greedy",

            "do_sample":
                False,

            "num_beams":
                1,

            "max_new_tokens":
                runner.MAX_NEW_TOKENS,

            "eos_token_ids":
                list(
                    runner.EOS_TOKEN_IDS
                ),

            "pad_token_id":
                runner.PAD_TOKEN_ID,

            "input_truncation_permitted":
                False,

            "retry_permitted":
                False,

            "llm_judge_used":
                False,
        },

        "evaluation_order": [
            "pinned_base",
            "same_base_plus_frozen_candidate_adapter",
        ],

        "base":
            base_result,

        "adapted":
            adapted_result,

        "paired":
            paired,

        "acceptance_gates":
            gates,

        "governance": {
            "greenhouse_consumed":
                True,

            "reexecution_permitted":
                False,

            "retry_permitted":
                False,

            "tuning_after_results_permitted":
                False,

            "candidate_replacement_permitted":
                False,

            "training_executed":
                False,

            "optimizer_created":
                False,

            "backward_executed":
                False,
        },
    }


    atomic_write_json_exclusive(
        path=
            report_path,
        payload=
            report,
    )


    report_sha256 = (
        sha256_file(
            report_path
        )
    )


    receipt = {
        "receipt_rule_version":
            "qlora_v0.4_greenhouse_final_acceptance_receipt_v0.1",

        "launcher_rule_version":
            GREENHOUSE_FINAL_ACCEPTANCE_LAUNCHER_RULE_VERSION,

        "experiment_id":
            EXPERIMENT_ID,

        "final_acceptance_revision":
            FINAL_ACCEPTANCE_REVISION,

        "holdout_id":
            marker[
                "holdout_id"
            ],

        "created_at":
            created_at,

        "git_head":
            report[
                "git_head"
            ],

        "status":
            "completed",

        "execution_authorization_sha256":
            authorization_sha256,

        "execution_manifest_sha256":
            safe[
                "manifest_sha256"
            ],

        "launcher_sha256":
            safe[
                "launcher_sha256"
            ],

        "runner_sha256":
            EXPECTED_RUNNER_SHA256,

        "consumption_marker_sha256":
            report[
                "consumption_marker_sha256"
            ],

        "report_sha256":
            report_sha256,

        "adapter_bundle_sha256":
            EXPECTED_ADAPTER_BUNDLE_SHA256,

        "base_accuracy":
            base_result[
                "accuracy"
            ],

        "adapted_accuracy":
            adapted_result[
                "accuracy"
            ],

        "accuracy_delta":
            paired[
                "accuracy_delta"
            ],

        "base_macro_accuracy":
            base_result[
                "macro_accuracy"
            ],

        "adapted_macro_accuracy":
            adapted_result[
                "macro_accuracy"
            ],

        "macro_accuracy_delta":
            paired[
                "macro_accuracy_delta"
            ],

        "adapted_strict_json_validity_rate":
            adapted_result[
                "strict_json_validity_rate"
            ],

        "dangerous_false_positives":
            adapted_result[
                "dangerous_false_positives"
            ],

        "safety_failures":
            adapted_result[
                "safety_failures"
            ],

        "all_greenhouse_gates_passed":
            gates[
                "all_passed"
            ],

        "failure_action":
            gates[
                "failure_action"
            ],

        "greenhouse_consumed":
            True,

        "reexecution_permitted":
            False,

        "retry_permitted":
            False,

        "training_executed":
            False,

        "optimizer_created":
            False,

        "backward_executed":
            False,
    }


    atomic_write_json_exclusive(
        path=
            receipt_path,
        payload=
            receipt,
    )


    print(
        "=== DATALENS QLORA v0.4 "
        "GREENHOUSE FINAL ACCEPTANCE v0.1 ==="
    )

    print(
        "Greenhouse holdout consumed: True"
    )

    print(
        "Single-use marker created: True"
    )

    print(
        f"Base accuracy:    {base_result['accuracy']:.6f}"
    )

    print(
        f"Adapted accuracy: {adapted_result['accuracy']:.6f}"
    )

    print(
        f"Accuracy delta:   {paired['accuracy_delta']:+.6f}"
    )

    print(
        (
            "Adapted strict JSON validity: "
            f"{adapted_result['strict_json_validity_rate']:.6f}"
        )
    )

    print(
        (
            "Dangerous false positives: "
            f"{adapted_result['dangerous_false_positives']}"
        )
    )

    print(
        (
            "Safety failures: "
            f"{adapted_result['safety_failures']}"
        )
    )

    print(
        (
            "All Greenhouse gates passed: "
            f"{gates['all_passed']}"
        )
    )

    print(
        f"Report SHA256:  {report_sha256}"
    )

    print(
        (
            "Receipt SHA256: "
            f"{sha256_file(receipt_path)}"
        )
    )

    print(
        (
            "Consumption marker SHA256: "
            f"{sha256_file(marker_path)}"
        )
    )

    print(
        "Greenhouse reexecution permitted: False"
    )

    print(
        "Training executed: False"
    )

    print(
        "Optimizer created: False"
    )

    print(
        "Backward executed: False"
    )

    print(
        "DATALENS QLORA v0.4 "
        "GREENHOUSE FINAL ACCEPTANCE v0.1: COMPLETED"
    )


    return report


def main() -> None:

    execute_greenhouse_final_acceptance()


if __name__ == "__main__":

    main()
