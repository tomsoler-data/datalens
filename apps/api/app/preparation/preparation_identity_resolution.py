from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from threading import RLock
from typing import Literal

from app.preparation.dataset_identity import (
    DatasetIdentityReport,
)


PREPARATION_IDENTITY_RESOLUTION_VERSION = (
    "preparation_identity_resolution_v0.1"
)


IdentityResolutionKind = Literal[
    "detected_key",
    "continued_without_surrogate",
]


@dataclass(frozen=True)
class PreparationIdentityResolution:
    workflow_id: str
    dataset_id: str
    request_id: str
    kind: IdentityResolutionKind
    rule_version: str = (
        PREPARATION_IDENTITY_RESOLUTION_VERSION
    )


class PreparationIdentityResolutionError(
    RuntimeError,
):
    pass


_LOCK = RLock()

_CONTINUATIONS: dict[
    tuple[str, str],
    PreparationIdentityResolution,
] = {}


def build_identity_resolution_request_id(
    *,
    workflow_id: str,
    dataset_id: str,
    dataset_filename: str,
    artifact_stage: str,
    report: DatasetIdentityReport,
) -> str:
    payload = {
        "workflow_id":
            workflow_id,

        "dataset_id":
            dataset_id,

        "dataset_filename":
            dataset_filename,

        "artifact_stage":
            artifact_stage,

        "identity_report":
            report.model_dump(
                mode="json"
            ),
    }

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    digest = hashlib.sha256(
        canonical.encode(
            "utf-8"
        )
    ).hexdigest()[
        :20
    ]

    return (
        f"identity-surrogate:{digest}"
    )


def get_current_identity_resolution(
    *,
    workflow_id: str,
    dataset_id: str,
    dataset_filename: str,
    artifact_stage: str,
    report: DatasetIdentityReport,
) -> PreparationIdentityResolution | None:
    if (
        report.status
        in {
            "single_key",
            "composite_key",
        }
    ):
        return (
            PreparationIdentityResolution(
                workflow_id=
                    workflow_id,

                dataset_id=
                    dataset_id,

                request_id=
                    build_identity_resolution_request_id(
                        workflow_id=
                            workflow_id,

                        dataset_id=
                            dataset_id,

                        dataset_filename=
                            dataset_filename,

                        artifact_stage=
                            artifact_stage,

                        report=
                            report,
                    ),

                kind=
                    "detected_key",
            )
        )

    expected_request_id = (
        build_identity_resolution_request_id(
            workflow_id=
                workflow_id,

            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            artifact_stage=
                artifact_stage,

            report=
                report,
        )
    )

    with _LOCK:
        stored = (
            _CONTINUATIONS.get(
                (
                    workflow_id,
                    dataset_id,
                )
            )
        )

    if (
        stored is None
        or
        stored.request_id
        !=
        expected_request_id
    ):
        return None

    return stored


def record_continue_without_surrogate(
    *,
    workflow_id: str,
    dataset_id: str,
    dataset_filename: str,
    artifact_stage: str,
    report: DatasetIdentityReport,
    request_id: str,
) -> PreparationIdentityResolution:
    if (
        not report
        .surrogate_key_recommended
    ):
        raise (
            PreparationIdentityResolutionError(
                (
                    "Continue-without-surrogate is only valid "
                    "when Python recommends a surrogate key."
                )
            )
        )

    expected_request_id = (
        build_identity_resolution_request_id(
            workflow_id=
                workflow_id,

            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            artifact_stage=
                artifact_stage,

            report=
                report,
        )
    )

    if (
        request_id.strip()
        !=
        expected_request_id
    ):
        raise (
            PreparationIdentityResolutionError(
                (
                    "Identity continuation approval is stale "
                    "or does not match the current deterministic "
                    "identity report."
                )
            )
        )

    resolution = (
        PreparationIdentityResolution(
            workflow_id=
                workflow_id,

            dataset_id=
                dataset_id,

            request_id=
                expected_request_id,

            kind=
                "continued_without_surrogate",
        )
    )

    with _LOCK:
        _CONTINUATIONS[
            (
                workflow_id,
                dataset_id,
            )
        ] = (
            resolution
        )

    return resolution


def clear_identity_resolution(
    *,
    workflow_id: str,
    dataset_id: str,
) -> None:
    with _LOCK:
        _CONTINUATIONS.pop(
            (
                workflow_id,
                dataset_id,
            ),
            None,
        )


def reset_preparation_identity_resolution_for_tests(
) -> None:
    with _LOCK:
        _CONTINUATIONS.clear()
