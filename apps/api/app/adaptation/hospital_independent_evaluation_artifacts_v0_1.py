from __future__ import annotations


import hashlib
import json
import os
import re


from collections.abc import (
    Mapping,
    Sequence,
)

from pathlib import Path
from typing import Any


HOSPITAL_INDEPENDENT_EVALUATION_ARTIFACT_RULE_VERSION = (
    "qlora_v0.4_hospital_independent_evaluation_artifacts_v0.1"
)

HOSPITAL_PREDICTIONS_RULE_VERSION = (
    "qlora_v0.4_hospital_independent_evaluation_predictions_v0.1"
)

HOSPITAL_REPORT_RULE_VERSION = (
    "qlora_v0.4_hospital_independent_evaluation_report_v0.1"
)

HOSPITAL_RECEIPT_RULE_VERSION = (
    "qlora_v0.4_hospital_independent_evaluation_receipt_v0.1"
)


EXPECTED_CASE_COUNT = 30
EXPECTED_DECIMAL_PLACES = 6

EXPECTED_RELATIONS = (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
)


EXPECTED_PROTOCOL_SHA256 = (
    "0e958d67a6294f8666485a6274b1eee2"
    "1b9f06fe2e300ba93241a7ffaba3127d"
)

EXPECTED_SCORER_SHA256 = (
    "9b9d648716f96f4a062ecb01e1e18b73"
    "a74aa0ff5e346a78048d0d7e77c8ab13"
)

EXPECTED_SCORER_TEST_SHA256 = (
    "55851a2fe2a51e7f9e18fe60768fb868"
    "1286adc03955f5131a82b9690541cc4d"
)

EXPECTED_SCORING_RULE_VERSION = (
    "qlora_v0.4_hospital_independent_evaluation_scoring_v0.1"
)


SOURCE_AUTHORITIES = {
    (
        "apps/api/app/adaptation/"
        "hospital_independent_evaluation_runner_v0_4_v0_1.py"
    ):
        (
            "4f1f69ff01b9a9c096dd42fc694a6f9"
            "f11f0ba30a23f6fc2a8dddc92deff900d"
        ),

    (
        "apps/api/app/adaptation/"
        "hospital_independent_evaluation_execution_v0_1.py"
    ):
        (
            "28566b79ad34e2587ced709f68ca19b1"
            "dfaa5d4074c07ba769db3a11306ee1c1"
        ),

    (
        "apps/api/app/adaptation/"
        "hospital_independent_evaluation_preflight_v0_1.py"
    ):
        (
            "bd9b55863b001e921c0cc398e2c9f7d7"
            "882eb111aa0fec07b231dfdcc5aca2a5"
        ),

    (
        "apps/api/app/adaptation/"
        "hospital_independent_evaluation_launch_core_v0_1.py"
    ):
        (
            "db387980ef930232c91f705bc683af073"
            "c3ac19def95d0ea363619e26e39c20c"
        ),

    (
        "apps/api/app/adaptation/"
        "hospital_independent_evaluation_scoring_v0_1.py"
    ):
        EXPECTED_SCORER_SHA256,

    (
        "apps/api/tests/adaptation/"
        "test_hospital_independent_evaluation_scoring_v0_1.py"
    ):
        EXPECTED_SCORER_TEST_SHA256,
}


CONSUMPTION_MARKER_FILENAME = (
    "datalens_semantic_qlora_v0.4_"
    "hospital_independent_evaluation_v0.1_consumption.json"
)

PREDICTIONS_FILENAME = (
    "datalens_semantic_qlora_v0.4_"
    "hospital_independent_evaluation_v0.1_predictions.json"
)

REPORT_FILENAME = (
    "datalens_semantic_qlora_v0.4_"
    "hospital_independent_evaluation_v0.1_report.json"
)

RECEIPT_FILENAME = (
    "datalens_semantic_qlora_v0.4_"
    "hospital_independent_evaluation_v0.1_receipt.json"
)


FORBIDDEN_PREDICTION_FIELDS = {
    "gold_relation",
    "gold_reason",
}


def canonical_json_bytes(
    payload: Any,
) -> bytes:

    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def sha256_bytes(
    payload: bytes,
) -> str:

    return hashlib.sha256(
        payload
    ).hexdigest()


def sha256_file(
    path: Path,
) -> str:

    return sha256_bytes(
        path.read_bytes()
    )


def _require_commit_sha(
    value: Any,
) -> str:

    if (
        not isinstance(
            value,
            str,
        )
        or
        re.fullmatch(
            r"[0-9a-f]{40}",
            value,
        )
        is None
    ):

        raise ValueError(
            (
                "execution_authority_commit must be a "
                "40-character lowercase Git SHA."
            )
        )

    return value


def _require_sha256(
    value: Any,
    *,
    label: str,
) -> str:

    if (
        not isinstance(
            value,
            str,
        )
        or
        re.fullmatch(
            r"[0-9a-f]{64}",
            value,
        )
        is None
    ):

        raise ValueError(
            f"{label} must be a lowercase SHA256."
        )

    return value


def _require_utc_timestamp(
    value: Any,
    *,
    label: str,
) -> str:

    if (
        not isinstance(
            value,
            str,
        )
        or
        not value.endswith(
            "Z"
        )
        or
        len(
            value
        )
        <
        20
    ):

        raise ValueError(
            f"{label} must be an explicit UTC timestamp."
        )

    return value


def _require_sequence(
    value: Any,
    *,
    label: str,
) -> Sequence[Any]:

    if (
        not isinstance(
            value,
            Sequence,
        )
        or
        isinstance(
            value,
            (
                str,
                bytes,
                bytearray,
            ),
        )
    ):

        raise TypeError(
            f"{label} must be a sequence."
        )

    return value


def _json_clone(
    value: Any,
) -> Any:

    return json.loads(
        canonical_json_bytes(
            value
        ).decode(
            "utf-8"
        )
    )


def single_use_artifact_paths(
    *,
    artifact_dir: Path,
) -> dict[
    str,
    Path,
]:

    artifact_dir = (
        artifact_dir.resolve()
    )

    return {
        "consumption_marker":
            artifact_dir
            /
            CONSUMPTION_MARKER_FILENAME,

        "predictions":
            artifact_dir
            /
            PREDICTIONS_FILENAME,

        "report":
            artifact_dir
            /
            REPORT_FILENAME,

        "receipt":
            artifact_dir
            /
            RECEIPT_FILENAME,
    }


def _normalize_prediction_pass(
    *,
    model_label: str,
    results: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> tuple[
    dict[
        str,
        Any,
    ],
    ...,
]:

    if model_label not in {
        "base",
        "adapted",
    }:

        raise ValueError(
            "model_label must be base or adapted."
        )

    sequence = _require_sequence(
        results,
        label=
            f"{model_label}_results",
    )

    if (
        len(
            sequence
        )
        !=
        EXPECTED_CASE_COUNT
    ):

        raise RuntimeError(
            (
                f"{model_label} results must contain "
                f"exactly {EXPECTED_CASE_COUNT} cases."
            )
        )

    required_fields = {
        "case_id",
        "model_label",
        "prompt_sha256",
        "strict_json_valid",
        "predicted_relation",
    }

    normalized = []
    seen_case_ids = set()

    for record in sequence:

        if not isinstance(
            record,
            Mapping,
        ):

            raise TypeError(
                (
                    f"{model_label} prediction "
                    "must be a mapping."
                )
            )

        record_keys = set(
            record
        )

        forbidden = (
            record_keys
            &
            FORBIDDEN_PREDICTION_FIELDS
        )

        if forbidden:

            raise RuntimeError(
                (
                    "Gold material crossed predictions "
                    f"boundary: {sorted(forbidden)}"
                )
            )

        missing = (
            required_fields
            -
            record_keys
        )

        if missing:

            raise RuntimeError(
                (
                    f"{model_label} prediction missing "
                    f"fields: {sorted(missing)}"
                )
            )

        case_id = record[
            "case_id"
        ]

        if (
            not isinstance(
                case_id,
                str,
            )
            or
            not case_id.strip()
        ):

            raise TypeError(
                "case_id must be a non-empty string."
            )

        if case_id in seen_case_ids:

            raise RuntimeError(
                (
                    f"Duplicate {model_label} "
                    f"case_id: {case_id}"
                )
            )

        seen_case_ids.add(
            case_id
        )

        if (
            record[
                "model_label"
            ]
            !=
            model_label
        ):

            raise RuntimeError(
                (
                    f"{model_label} result has "
                    "incorrect model_label."
                )
            )

        _require_sha256(
            record[
                "prompt_sha256"
            ],
            label=
                "prompt_sha256",
        )

        strict_json_valid = record[
            "strict_json_valid"
        ]

        if not isinstance(
            strict_json_valid,
            bool,
        ):

            raise TypeError(
                "strict_json_valid must be boolean."
            )

        predicted_relation = record[
            "predicted_relation"
        ]

        if strict_json_valid:

            if (
                predicted_relation
                not in
                EXPECTED_RELATIONS
            ):

                raise RuntimeError(
                    (
                        "Strict-valid result has invalid "
                        "predicted_relation."
                    )
                )

        elif (
            predicted_relation
            is not None
        ):

            raise RuntimeError(
                (
                    "Strict-invalid result must have "
                    "predicted_relation=None."
                )
            )

        normalized.append(
            _json_clone(
                dict(
                    record
                )
            )
        )

    return tuple(
        normalized
    )


def _validate_base_adapted_identity(
    *,
    base_results: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    adapted_results: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> None:

    for (
        base,
        adapted,
    ) in zip(
        base_results,
        adapted_results,
        strict=True,
    ):

        if (
            base[
                "case_id"
            ]
            !=
            adapted[
                "case_id"
            ]
        ):

            raise RuntimeError(
                "Base/Adapted case ordering changed."
            )

        if (
            base[
                "prompt_sha256"
            ]
            !=
            adapted[
                "prompt_sha256"
            ]
        ):

            raise RuntimeError(
                "Base/Adapted prompt identity changed."
            )


def build_predictions_artifact(
    *,
    execution_authority_commit: str,
    claimed_at_utc: str,
    base_results: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    adapted_results: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    Any,
]:

    execution_authority_commit = (
        _require_commit_sha(
            execution_authority_commit
        )
    )

    claimed_at_utc = (
        _require_utc_timestamp(
            claimed_at_utc,
            label=
                "claimed_at_utc",
        )
    )

    base = _normalize_prediction_pass(
        model_label=
            "base",
        results=
            base_results,
    )

    adapted = _normalize_prediction_pass(
        model_label=
            "adapted",
        results=
            adapted_results,
    )

    _validate_base_adapted_identity(
        base_results=
            base,
        adapted_results=
            adapted,
    )

    return {
        "rule_version":
            HOSPITAL_PREDICTIONS_RULE_VERSION,

        "execution_authority_commit":
            execution_authority_commit,

        "claimed_at_utc":
            claimed_at_utc,

        "case_count":
            EXPECTED_CASE_COUNT,

        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA256,

        "scorer_sha256":
            EXPECTED_SCORER_SHA256,

        "base_results":
            list(
                base
            ),

        "adapted_results":
            list(
                adapted
            ),
    }


def _validate_scoring_result(
    scoring_result: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:

    if not isinstance(
        scoring_result,
        Mapping,
    ):

        raise TypeError(
            "scoring_result must be a mapping."
        )

    expected_keys = {
        "rule_version",
        "case_count",
        "decimal_places",
        "base",
        "adapted",
        "comparison",
        "gates",
        "promotion_eligible",
    }

    if (
        set(
            scoring_result
        )
        !=
        expected_keys
    ):

        raise RuntimeError(
            "Scoring result key set changed."
        )

    if (
        scoring_result[
            "rule_version"
        ]
        !=
        EXPECTED_SCORING_RULE_VERSION
    ):

        raise RuntimeError(
            "Scoring rule version changed."
        )

    if (
        scoring_result[
            "case_count"
        ]
        !=
        EXPECTED_CASE_COUNT
    ):

        raise RuntimeError(
            "Scoring case count changed."
        )

    if (
        scoring_result[
            "decimal_places"
        ]
        !=
        EXPECTED_DECIMAL_PLACES
    ):

        raise RuntimeError(
            "Scoring decimal precision changed."
        )

    for (
        key,
        expected_model_label,
    ) in (
        (
            "base",
            "base",
        ),
        (
            "adapted",
            "adapted",
        ),
    ):

        section = scoring_result[
            key
        ]

        if not isinstance(
            section,
            Mapping,
        ):

            raise TypeError(
                f"{key} score must be a mapping."
            )

        if (
            section.get(
                "model_label"
            )
            !=
            expected_model_label
        ):

            raise RuntimeError(
                f"{key} score model_label changed."
            )

    comparison = scoring_result[
        "comparison"
    ]

    if not isinstance(
        comparison,
        Mapping,
    ):

        raise TypeError(
            "comparison must be a mapping."
        )

    if (
        set(
            comparison
        )
        !=
        {
            "accuracy_delta",
            "macro_accuracy_delta",
            "correct_case_count_delta",
        }
    ):

        raise RuntimeError(
            "Comparison metric contract changed."
        )

    gates = scoring_result[
        "gates"
    ]

    if not isinstance(
        gates,
        Mapping,
    ):

        raise TypeError(
            "gates must be a mapping."
        )

    if (
        set(
            gates
        )
        !=
        {
            "adapted_absolute",
            "all_absolute_gates_pass",
            "non_regression",
            "all_non_regression_gates_pass",
            "promotion_signal",
            "promotion_signal_pass",
        }
    ):

        raise RuntimeError(
            "Gate result contract changed."
        )

    if not isinstance(
        scoring_result[
            "promotion_eligible"
        ],
        bool,
    ):

        raise TypeError(
            "promotion_eligible must be boolean."
        )

    return _json_clone(
        dict(
            scoring_result
        )
    )


def build_report_artifact(
    *,
    execution_authority_commit: str,
    claimed_at_utc: str,
    scoring_result: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:

    execution_authority_commit = (
        _require_commit_sha(
            execution_authority_commit
        )
    )

    claimed_at_utc = (
        _require_utc_timestamp(
            claimed_at_utc,
            label=
                "claimed_at_utc",
        )
    )

    scoring = _validate_scoring_result(
        scoring_result
    )

    return {
        "rule_version":
            HOSPITAL_REPORT_RULE_VERSION,

        "execution_authority_commit":
            execution_authority_commit,

        "claimed_at_utc":
            claimed_at_utc,

        "case_count":
            EXPECTED_CASE_COUNT,

        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA256,

        "scorer_sha256":
            EXPECTED_SCORER_SHA256,

        "scoring":
            scoring,

        "promotion_eligible":
            scoring[
                "promotion_eligible"
            ],
    }


def _validate_consumption_marker(
    *,
    marker_path: Path,
    execution_authority_commit: str,
    claimed_at_utc: str,
) -> dict[
    str,
    Any,
]:

    if not marker_path.is_file():

        raise RuntimeError(
            "Consumption marker must already exist."
        )

    try:

        marker = json.loads(
            marker_path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:

        raise RuntimeError(
            "Consumption marker cannot be validated."
        ) from exc

    if not isinstance(
        marker,
        dict,
    ):

        raise RuntimeError(
            "Consumption marker must be a JSON object."
        )

    if (
        marker.get(
            "status"
        )
        !=
        "claimed_before_holdout_open"
    ):

        raise RuntimeError(
            "Consumption marker status changed."
        )

    if (
        marker.get(
            "execution_authority_commit"
        )
        !=
        execution_authority_commit
    ):

        raise RuntimeError(
            (
                "Consumption marker commit does not "
                "match execution."
            )
        )

    if (
        marker.get(
            "claimed_at_utc"
        )
        !=
        claimed_at_utc
    ):

        raise RuntimeError(
            (
                "Consumption marker timestamp does not "
                "match execution."
            )
        )

    return marker


def build_receipt_artifact(
    *,
    execution_authority_commit: str,
    claimed_at_utc: str,
    completed_at_utc: str,
    consumption_marker_sha256: str,
    predictions_sha256: str,
    report_sha256: str,
    promotion_eligible: bool,
) -> dict[
    str,
    Any,
]:

    execution_authority_commit = (
        _require_commit_sha(
            execution_authority_commit
        )
    )

    claimed_at_utc = (
        _require_utc_timestamp(
            claimed_at_utc,
            label=
                "claimed_at_utc",
        )
    )

    completed_at_utc = (
        _require_utc_timestamp(
            completed_at_utc,
            label=
                "completed_at_utc",
        )
    )

    consumption_marker_sha256 = (
        _require_sha256(
            consumption_marker_sha256,
            label=
                "consumption_marker_sha256",
        )
    )

    predictions_sha256 = (
        _require_sha256(
            predictions_sha256,
            label=
                "predictions_sha256",
        )
    )

    report_sha256 = (
        _require_sha256(
            report_sha256,
            label=
                "report_sha256",
        )
    )

    if not isinstance(
        promotion_eligible,
        bool,
    ):

        raise TypeError(
            "promotion_eligible must be boolean."
        )

    return {
        "rule_version":
            HOSPITAL_RECEIPT_RULE_VERSION,

        "status":
            "completed",

        "execution_authority_commit":
            execution_authority_commit,

        "claimed_at_utc":
            claimed_at_utc,

        "completed_at_utc":
            completed_at_utc,

        "case_count":
            EXPECTED_CASE_COUNT,

        "protocol_sha256":
            EXPECTED_PROTOCOL_SHA256,

        "scorer_sha256":
            EXPECTED_SCORER_SHA256,

        "source_authorities":
            dict(
                SOURCE_AUTHORITIES
            ),

        "consumption_marker_sha256":
            consumption_marker_sha256,

        "predictions_sha256":
            predictions_sha256,

        "report_sha256":
            report_sha256,

        "promotion_eligible":
            promotion_eligible,
    }


def _atomic_write_bytes_exclusive(
    *,
    path: Path,
    payload: bytes,
) -> None:

    path = path.resolve()

    if not path.parent.is_dir():

        raise RuntimeError(
            (
                "Artifact directory must already exist: "
                f"{path.parent}"
            )
        )

    flags = (
        os.O_WRONLY
        |
        os.O_CREAT
        |
        os.O_EXCL
    )

    try:

        descriptor = os.open(
            str(
                path
            ),
            flags,
            0o600,
        )

    except FileExistsError as exc:

        raise RuntimeError(
            (
                "Exclusive Hospital artifact already "
                f"exists: {path}"
            )
        ) from exc

    try:

        with os.fdopen(
            descriptor,
            "wb",
        ) as handle:

            handle.write(
                payload
            )

            handle.flush()

            os.fsync(
                handle.fileno()
            )

    except BaseException:

        # Intentionally fail closed.
        #
        # Once exclusive creation succeeds, the file is never
        # deleted or rewritten to make the benchmark appear
        # unused or incomplete again.
        raise


def _require_chain_available(
    *,
    paths: Mapping[
        str,
        Path,
    ],
) -> None:

    marker = paths[
        "consumption_marker"
    ]

    if not marker.is_file():

        raise RuntimeError(
            "Consumption marker does not exist."
        )

    existing_outputs = [
        (
            label,
            paths[
                label
            ],
        )

        for label in (
            "predictions",
            "report",
            "receipt",
        )

        if paths[
            label
        ].exists()
    ]

    if existing_outputs:

        details = ", ".join(
            (
                f"{label}={path}"
            )

            for (
                label,
                path,
            )
            in existing_outputs
        )

        raise RuntimeError(
            (
                "Hospital official artifact chain is no "
                "longer available for first write: "
                f"{details}"
            )
        )


def write_official_artifact_chain(
    *,
    artifact_dir: Path,
    execution_authority_commit: str,
    claimed_at_utc: str,
    completed_at_utc: str,
    base_results: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    adapted_results: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    scoring_result: Mapping[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:

    artifact_dir = (
        artifact_dir.resolve()
    )

    if not artifact_dir.is_dir():

        raise RuntimeError(
            "Artifact directory must already exist."
        )

    execution_authority_commit = (
        _require_commit_sha(
            execution_authority_commit
        )
    )

    claimed_at_utc = (
        _require_utc_timestamp(
            claimed_at_utc,
            label=
                "claimed_at_utc",
        )
    )

    completed_at_utc = (
        _require_utc_timestamp(
            completed_at_utc,
            label=
                "completed_at_utc",
        )
    )

    paths = single_use_artifact_paths(
        artifact_dir=
            artifact_dir
    )

    _require_chain_available(
        paths=
            paths
    )

    _validate_consumption_marker(
        marker_path=
            paths[
                "consumption_marker"
            ],
        execution_authority_commit=
            execution_authority_commit,
        claimed_at_utc=
            claimed_at_utc,
    )

    predictions = build_predictions_artifact(
        execution_authority_commit=
            execution_authority_commit,
        claimed_at_utc=
            claimed_at_utc,
        base_results=
            base_results,
        adapted_results=
            adapted_results,
    )

    report = build_report_artifact(
        execution_authority_commit=
            execution_authority_commit,
        claimed_at_utc=
            claimed_at_utc,
        scoring_result=
            scoring_result,
    )

    predictions_bytes = canonical_json_bytes(
        predictions
    )

    report_bytes = canonical_json_bytes(
        report
    )

    consumption_marker_sha = sha256_file(
        paths[
            "consumption_marker"
        ]
    )

    predictions_sha = sha256_bytes(
        predictions_bytes
    )

    report_sha = sha256_bytes(
        report_bytes
    )

    receipt = build_receipt_artifact(
        execution_authority_commit=
            execution_authority_commit,
        claimed_at_utc=
            claimed_at_utc,
        completed_at_utc=
            completed_at_utc,
        consumption_marker_sha256=
            consumption_marker_sha,
        predictions_sha256=
            predictions_sha,
        report_sha256=
            report_sha,
        promotion_eligible=
            report[
                "promotion_eligible"
            ],
    )

    receipt_bytes = canonical_json_bytes(
        receipt
    )

    receipt_sha = sha256_bytes(
        receipt_bytes
    )

    # Official single-use write order:
    #
    #   1. predictions
    #   2. report
    #   3. receipt LAST
    #
    # No cleanup occurs after successful exclusive creation.

    _atomic_write_bytes_exclusive(
        path=
            paths[
                "predictions"
            ],
        payload=
            predictions_bytes,
    )

    _atomic_write_bytes_exclusive(
        path=
            paths[
                "report"
            ],
        payload=
            report_bytes,
    )

    _atomic_write_bytes_exclusive(
        path=
            paths[
                "receipt"
            ],
        payload=
            receipt_bytes,
    )

    return {
        "rule_version":
            HOSPITAL_INDEPENDENT_EVALUATION_ARTIFACT_RULE_VERSION,

        "status":
            "completed",

        "execution_authority_commit":
            execution_authority_commit,

        "predictions_path":
            str(
                paths[
                    "predictions"
                ]
            ),

        "predictions_sha256":
            predictions_sha,

        "report_path":
            str(
                paths[
                    "report"
                ]
            ),

        "report_sha256":
            report_sha,

        "receipt_path":
            str(
                paths[
                    "receipt"
                ]
            ),

        "receipt_sha256":
            receipt_sha,

        "promotion_eligible":
            report[
                "promotion_eligible"
            ],
    }
