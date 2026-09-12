from __future__ import annotations


from collections import Counter
from datetime import datetime, timezone
from collections.abc import (
    Mapping,
    Sequence,
)

from pathlib import Path
from typing import Any


from app.adaptation.greenhouse_final_acceptance_runner_v0_4_v0_4 import (
    attach_adapter,
    load_base_model,
    prepare_runtime_authority,
)

from app.adaptation.hospital_independent_evaluation_artifacts_v0_1 import (
    write_official_artifact_chain,
)

from app.adaptation.hospital_independent_evaluation_execution_v0_1 import (
    run_model_pass,
)

from app.adaptation.hospital_independent_evaluation_launch_core_v0_1 import (
    run_official_launch_core,
)

from app.adaptation.hospital_independent_evaluation_preflight_v0_1 import (
    build_preflight_snapshot,
    claim_single_use_consumption,
)

from app.adaptation.hospital_independent_evaluation_runner_v0_4_v0_1 import (
    build_execution_inputs,
    load_frozen_holdout_for_execution,
)

from app.adaptation.hospital_independent_evaluation_scoring_v0_1 import (
    score_official_comparison,
)


HOSPITAL_INDEPENDENT_EVALUATION_ORCHESTRATOR_RULE_VERSION = (
    "qlora_v0.4_hospital_independent_evaluation_orchestrator_v0.1"
)


EXPECTED_CASE_COUNT = 30
EXPECTED_CASES_PER_RELATION = 6

EXPECTED_RELATIONS = (
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
)


GREENHOUSE_RUNTIME_COMMITTED_SHA256 = (
    "8e922f46d65048ab6bcebeab8eca7e52"
    "b8bf83100524bea84658530039b40cea"
)


class _GoldProjectionCapture:

    def __init__(
        self,
    ) -> None:

        self._records: (
            tuple[
                dict[
                    str,
                    str,
                ],
                ...,
            ]
            |
            None
        ) = None


    def capture(
        self,
        cases: Sequence[
            Mapping[
                str,
                Any,
            ]
        ],
    ) -> None:

        if self._records is not None:

            raise RuntimeError(
                "Hospital gold projection was captured more than once."
            )

        if (
            not isinstance(
                cases,
                Sequence,
            )
            or
            isinstance(
                cases,
                (
                    str,
                    bytes,
                    bytearray,
                ),
            )
        ):

            raise TypeError(
                "Protected cases must be a sequence."
            )

        if (
            len(
                cases
            )
            !=
            EXPECTED_CASE_COUNT
        ):

            raise RuntimeError(
                (
                    "Hospital holdout projection must contain "
                    f"exactly {EXPECTED_CASE_COUNT} cases."
                )
            )

        records = []
        seen_case_ids = set()
        relation_counts = Counter()

        for case in cases:

            if not isinstance(
                case,
                Mapping,
            ):

                raise TypeError(
                    "Protected case must be a mapping."
                )

            case_id = case.get(
                "case_id"
            )

            relation = case.get(
                "gold_relation"
            )

            if (
                not isinstance(
                    case_id,
                    str,
                )
                or
                not case_id.strip()
            ):

                raise TypeError(
                    "Protected case_id must be a non-empty string."
                )

            if case_id in seen_case_ids:

                raise RuntimeError(
                    (
                        "Duplicate Hospital case_id during "
                        f"gold projection: {case_id}"
                    )
                )

            if relation not in EXPECTED_RELATIONS:

                raise RuntimeError(
                    (
                        "Unexpected Hospital gold relation: "
                        f"{relation!r}"
                    )
                )

            seen_case_ids.add(
                case_id
            )

            relation_counts[
                relation
            ] += 1

            records.append(
                {
                    "case_id":
                        case_id,

                    "gold_relation":
                        relation,
                }
            )

        for relation in EXPECTED_RELATIONS:

            if (
                relation_counts[
                    relation
                ]
                !=
                EXPECTED_CASES_PER_RELATION
            ):

                raise RuntimeError(
                    (
                        "Hospital gold projection balance changed: "
                        f"{relation}"
                    )
                )

        self._records = tuple(
            records
        )


    def require_projection(
        self,
    ) -> tuple[
        dict[
            str,
            str,
        ],
        ...,
    ]:

        if self._records is None:

            raise RuntimeError(
                (
                    "Hospital gold projection is unavailable. "
                    "Protected loader did not complete."
                )
            )

        return tuple(
            dict(
                record
            )

            for record
            in self._records
        )


def _utc_now(
) -> str:

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat(
            timespec="seconds"
        )
        .replace(
            "+00:00",
            "Z",
        )
    )


def _require_artifact_dir(
    artifact_dir: Path,
) -> Path:

    if not isinstance(
        artifact_dir,
        Path,
    ):

        raise TypeError(
            "artifact_dir must be a pathlib.Path."
        )

    resolved = artifact_dir.resolve()

    if not resolved.is_dir():

        raise RuntimeError(
            (
                "Hospital evaluation artifact directory "
                f"does not exist: {resolved}"
            )
        )

    return resolved


def _build_official_dependencies(
    *,
    artifact_dir: Path,
    gold_capture: _GoldProjectionCapture,
) -> dict[
    str,
    Any,
]:

    artifact_dir = _require_artifact_dir(
        artifact_dir
    )

    if not isinstance(
        gold_capture,
        _GoldProjectionCapture,
    ):

        raise TypeError(
            "gold_capture must be _GoldProjectionCapture."
        )


    def bound_preflight_snapshot(
    ) -> dict[
        str,
        Any,
    ]:

        return build_preflight_snapshot(
            artifact_dir=
                artifact_dir
        )


    def bound_consumption_claim(
        *,
        execution_authority_commit: str,
        claimed_at_utc: str,
    ) -> dict[
        str,
        Any,
    ]:

        return claim_single_use_consumption(
            execution_authority_commit=
                execution_authority_commit,

            claimed_at_utc=
                claimed_at_utc,

            artifact_dir=
                artifact_dir,
        )


    def bound_holdout_loader(
    ) -> tuple[
        dict[
            str,
            Any,
        ],
        ...,
    ]:

        protected_cases = (
            load_frozen_holdout_for_execution()
        )

        gold_capture.capture(
            protected_cases
        )

        return protected_cases


    return {
        "build_preflight_snapshot":
            bound_preflight_snapshot,

        "prepare_runtime_authority":
            prepare_runtime_authority,

        "load_base_model":
            load_base_model,

        "claim_single_use_consumption":
            bound_consumption_claim,

        "load_frozen_holdout_for_execution":
            bound_holdout_loader,

        "build_execution_inputs":
            build_execution_inputs,

        "run_model_pass":
            run_model_pass,

        "attach_adapter":
            attach_adapter,
    }


def _validate_launch_result(
    *,
    launch_result: Mapping[
        str,
        Any,
    ],
    execution_authority_commit: str,
    claimed_at_utc: str,
) -> dict[
    str,
    Any,
]:

    if not isinstance(
        launch_result,
        Mapping,
    ):

        raise TypeError(
            "Official launch result must be a mapping."
        )

    expected_keys = {
        "rule_version",
        "execution_authority_commit",
        "claimed_at_utc",
        "case_count",
        "preflight_rule_version",
        "consumption_claim",
        "base_results",
        "adapted_results",
        "scoring_started",
        "report_written",
        "receipt_written",
    }

    if (
        set(
            launch_result
        )
        !=
        expected_keys
    ):

        raise RuntimeError(
            "Official launch result key set changed."
        )

    if (
        launch_result[
            "execution_authority_commit"
        ]
        !=
        execution_authority_commit
    ):

        raise RuntimeError(
            "Launch execution authority changed."
        )

    if (
        launch_result[
            "claimed_at_utc"
        ]
        !=
        claimed_at_utc
    ):

        raise RuntimeError(
            "Launch claim timestamp changed."
        )

    if (
        launch_result[
            "case_count"
        ]
        !=
        EXPECTED_CASE_COUNT
    ):

        raise RuntimeError(
            "Launch case count changed."
        )

    for flag in (
        "scoring_started",
        "report_written",
        "receipt_written",
    ):

        if (
            launch_result[
                flag
            ]
            is not False
        ):

            raise RuntimeError(
                (
                    "Frozen launch core unexpectedly crossed "
                    f"post-generation boundary: {flag}"
                )
            )

    consumption_claim = launch_result[
        "consumption_claim"
    ]

    if not isinstance(
        consumption_claim,
        Mapping,
    ):

        raise TypeError(
            "Launch consumption claim must be a mapping."
        )

    if (
        consumption_claim.get(
            "status"
        )
        !=
        "claimed_before_holdout_open"
    ):

        raise RuntimeError(
            "Launch consumption claim status changed."
        )

    if (
        consumption_claim.get(
            "execution_authority_commit"
        )
        !=
        execution_authority_commit
    ):

        raise RuntimeError(
            "Launch consumption claim commit changed."
        )

    if (
        consumption_claim.get(
            "claimed_at_utc"
        )
        !=
        claimed_at_utc
    ):

        raise RuntimeError(
            "Launch consumption claim timestamp changed."
        )

    return dict(
        launch_result
    )


def run_official_evaluation(
    *,
    execution_authority_commit: str,
    claimed_at_utc: str,
    artifact_dir: Path,
    torch_module: Any,
) -> dict[
    str,
    Any,
]:

    artifact_dir = _require_artifact_dir(
        artifact_dir
    )

    gold_capture = (
        _GoldProjectionCapture()
    )

    dependencies = (
        _build_official_dependencies(
            artifact_dir=
                artifact_dir,

            gold_capture=
                gold_capture,
        )
    )

    launch_result = (
        _validate_launch_result(
            launch_result=
                run_official_launch_core(
                    execution_authority_commit=
                        execution_authority_commit,

                    claimed_at_utc=
                        claimed_at_utc,

                    dependencies=
                        dependencies,

                    torch_module=
                        torch_module,
                ),

            execution_authority_commit=
                execution_authority_commit,

            claimed_at_utc=
                claimed_at_utc,
        )
    )

    # This projection becomes accessible to scoring only after
    # the frozen launch core has completed BOTH Base and Adapted
    # passes successfully.
    gold_records = (
        gold_capture.require_projection()
    )

    scoring_result = (
        score_official_comparison(
            gold_records=
                gold_records,

            base_results=
                launch_result[
                    "base_results"
                ],

            adapted_results=
                launch_result[
                    "adapted_results"
                ],
        )
    )

    # Evaluation completion time is owned by the
    # orchestrator and is created only AFTER both model
    # passes and deterministic scoring have completed.
    completed_at_utc = (
        _utc_now()
    )

    artifact_result = (
        write_official_artifact_chain(
            artifact_dir=
                artifact_dir,

            execution_authority_commit=
                execution_authority_commit,

            claimed_at_utc=
                claimed_at_utc,

            completed_at_utc=
                completed_at_utc,

            base_results=
                launch_result[
                    "base_results"
                ],

            adapted_results=
                launch_result[
                    "adapted_results"
                ],

            scoring_result=
                scoring_result,
        )
    )

    if not isinstance(
        artifact_result,
        Mapping,
    ):

        raise TypeError(
            "Official artifact result must be a mapping."
        )

    if (
        artifact_result.get(
            "status"
        )
        !=
        "completed"
    ):

        raise RuntimeError(
            "Official artifact chain did not complete."
        )

    if (
        artifact_result.get(
            "execution_authority_commit"
        )
        !=
        execution_authority_commit
    ):

        raise RuntimeError(
            "Artifact execution authority changed."
        )

    return {
        "rule_version":
            HOSPITAL_INDEPENDENT_EVALUATION_ORCHESTRATOR_RULE_VERSION,

        "execution_authority_commit":
            execution_authority_commit,

        "claimed_at_utc":
            claimed_at_utc,

        "completed_at_utc":
            completed_at_utc,

        "case_count":
            EXPECTED_CASE_COUNT,

        "launch_core_rule_version":
            launch_result[
                "rule_version"
            ],

        "preflight_rule_version":
            launch_result[
                "preflight_rule_version"
            ],

        "scoring_rule_version":
            scoring_result.get(
                "rule_version"
            ),

        "promotion_eligible":
            scoring_result.get(
                "promotion_eligible"
            ),

        "artifact_result":
            dict(
                artifact_result
            ),
    }
