from __future__ import annotations


import hashlib

from typing import (
    Any,
)


from app.reporting.analysis_artifact_store import (
    AnalysisArtifactRecord,
    AnalysisSourceType,
    register_server_owned_analysis,
    build_server_owned_analysis_record,
    persist_server_owned_analysis_records_atomic,
)


# ============================================================
# VERSION
# ============================================================

UNIFIED_REPORT_ARTIFACT_RULE_VERSION = (
    "unified_report_artifacts_v0.2"
)


REQUESTED_ANALYSIS_SOURCE_TYPES = (
    frozenset(
        {
            "document_request",
            "follow_up_prompt",
        }
    )
)


def _require_requested_analysis_source_type(
    source_type: AnalysisSourceType,
) -> AnalysisSourceType:
    if (
        source_type
        not in
        REQUESTED_ANALYSIS_SOURCE_TYPES
    ):
        raise ValueError(
            (
                "Requested Analysis artifacts support only "
                "document_request or follow_up_prompt "
                "source types."
            )
        )

    return source_type


# ============================================================
# HELPERS
# ============================================================

def _dump(
    value: Any,
) -> dict[
    str,
    Any,
]:
    if hasattr(
        value,
        "model_dump",
    ):
        return (
            value.model_dump(
                mode="json"
            )
        )


    if isinstance(
        value,
        dict,
    ):
        return dict(
            value
        )


    raise TypeError(
        (
            "Unified report artifact registration "
            "requires Pydantic models or mappings."
        )
    )


def _stable_id(
    *,
    workflow_id: str,
    source_type: AnalysisSourceType,
    source_analysis_id: str,
) -> str:
    digest = (
        hashlib.sha256(
            (
                f"{workflow_id}\x1f"
                f"{source_type}\x1f"
                f"{source_analysis_id}"
            )
            .encode(
                "utf-8"
            )
        )
        .hexdigest()[
            :
            24
        ]
    )


    return (
        f"analysis:report:{digest}"
    )


def _string_list(
    value: object,
) -> list[
    str
]:
    if not isinstance(
        value,
        list,
    ):
        return []


    return [
        str(
            item
        )

        for item
        in value

        if str(
            item
        ).strip()
    ]


def _variables(
    finding: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    str,
]:
    raw = (
        finding.get(
            "variables"
        )
    )


    if not isinstance(
        raw,
        dict,
    ):
        return {}


    return {
        str(
            role
        ):
            str(
                column
            )

        for (
            role,
            column,
        )
        in raw.items()

        if (
            str(
                role
            ).strip()
            and
            str(
                column
            ).strip()
        )
    }


def _bindings(
    finding: dict[
        str,
        Any,
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    dataset_id = (
        finding.get(
            "dataset_id"
        )
    )


    dataset_filename = (
        finding.get(
            "dataset_filename"
        )
    )


    return [
        {
            "role":
                role,

            "column":
                column,

            "dataset_id":
                (
                    str(
                        dataset_id
                    )
                    if (
                        dataset_id
                        is not None
                    )
                    else
                    None
                ),

            "dataset_filename":
                (
                    str(
                        dataset_filename
                    )
                    if (
                        dataset_filename
                        is not None
                    )
                    else
                    None
                ),

            "semantic_concept":
                None,

            "analysis_kind":
                (
                    str(
                        finding.get(
                            "family",
                            "",
                        )
                    )
                    or
                    None
                ),
        }

        for (
            role,
            column,
        )
        in _variables(
            finding
        ).items()
    ]


def _requested_tool_name(
    family: str,
) -> str:
    normalized = (
        family
        .strip()
        .lower()
    )


    if not (
        normalized
    ):
        return (
            "run_report_finding"
        )


    return (
        "run_"
        +
        normalized
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )


def _synthetic_native_payload(
    *,
    artifact_id: str,
    source_type: AnalysisSourceType,
    objective: str,
    finding: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    family = str(
        finding.get(
            "family",
            "",
        )
        or
        "analysis"
    )


    source_analysis_id = str(
        finding.get(
            "analysis_id",
            "",
        )
        or
        artifact_id
    )


    title = str(
        finding.get(
            "title",
            "",
        )
        or
        objective
    )


    chart_type = str(
        finding.get(
            "chart_type",
            "",
        )
        or
        "none"
    )


    summary = (
        _string_list(
            finding.get(
                "summary"
            )
        )
    )


    reasons = (
        _string_list(
            finding.get(
                "reasons"
            )
        )
    )


    caveats = (
        _string_list(
            finding.get(
                "caveats"
            )
        )
    )


    datasets = (
        _string_list(
            finding.get(
                "datasets"
            )
        )
    )


    dataset_id = (
        finding.get(
            "dataset_id"
        )
    )


    if (
        dataset_id is not None
        and
        str(
            dataset_id
        ).strip()
        and
        str(
            dataset_id
        )
        not in datasets
    ):
        datasets.insert(
            0,
            str(
                dataset_id
            ),
        )


    contract_id = (
        f"report-contract:{source_analysis_id}"
    )


    contract = {
        "contract_id":
            contract_id,

        "contract_version":
            "report-finding-v0.1",

        "origin":
            (
                "document"
                if source_type
                ==
                "document_request"
                else
                "automatic"
            ),

        "status":
            "validated",

        "title":
            title,

        "request_text":
            objective,

        "family":
            family,

        "required_dataset_ids":
            datasets,

        "required_dataset_filenames":
            [],

        "analytical_grain":
            finding.get(
                "analytical_grain"
            ),

        "bindings":
            _bindings(
                finding
            ),

        "reasons":
            reasons,

        "blockers":
            [],

        "planner_confidence":
            1.0,
    }


    result = {
        "analysis_id":
            source_analysis_id,

        "dataset_id":
            (
                str(
                    dataset_id
                )
                if dataset_id
                is not None
                else
                None
            ),

        "dataset_filename":
            (
                str(
                    finding.get(
                        "dataset_filename"
                    )
                )
                if finding.get(
                    "dataset_filename"
                )
                is not None
                else
                None
            ),

        "title":
            title,

        "family":
            family,

        "execution_status":
            str(
                finding.get(
                    "execution_status",
                    "complete",
                )
                or
                "complete"
            ),

        "chart_type":
            chart_type,

        "summary":
            summary,

        "metrics":
            (
                dict(
                    finding.get(
                        "metrics"
                    )
                )
                if isinstance(
                    finding.get(
                        "metrics"
                    ),
                    dict,
                )
                else
                {}
            ),

        "chart_data":
            (
                list(
                    finding.get(
                        "chart_data"
                    )
                )
                if isinstance(
                    finding.get(
                        "chart_data"
                    ),
                    list,
                )
                else
                []
            ),

        "statistical_decision":
            finding.get(
                "statistical_decision"
            ),

        "statistical_result":
            finding.get(
                "statistical_result"
            ),

        "warnings":
            caveats,

        "limitations":
            [],

        "execution_rule_version":
            UNIFIED_REPORT_ARTIFACT_RULE_VERSION,
    }


    return {
        "analysis_id":
            artifact_id,

        "analysis_source_type":
            source_type,

        "status":
            "ready",

        "trace_id":
            f"report:{source_analysis_id}",

        "planner_model":
            "python-deterministic",

        "tool_model":
            "python-deterministic",

        "planner":
            {
                "status":
                    "ready",

                "objective":
                    objective,

                "model":
                    "python-deterministic",

                "proposal_count":
                    1,

                "validated_count":
                    1,

                "blocked_count":
                    0,

                "ambiguous_count":
                    0,

                "rejected_count":
                    0,

                "items":
                    [
                        {
                            "proposal_index":
                                0,

                            "validation_status":
                                "validated",

                            "proposal":
                                {
                                    "decision":
                                        "execute",

                                    "title":
                                        title,

                                    "family":
                                        family,

                                    "dataset_id":
                                        (
                                            str(
                                                dataset_id
                                            )
                                            if dataset_id
                                            is not None
                                            else
                                            None
                                        ),

                                    "analytical_grain":
                                        finding.get(
                                            "analytical_grain"
                                        ),

                                    "x_column":
                                        _variables(
                                            finding
                                        ).get(
                                            "x"
                                        ),

                                    "y_column":
                                        _variables(
                                            finding
                                        ).get(
                                            "y"
                                        ),

                                    "group_column":
                                        _variables(
                                            finding
                                        ).get(
                                            "group"
                                        ),

                                    "value_column":
                                        _variables(
                                            finding
                                        ).get(
                                            "value"
                                        ),

                                    "time_column":
                                        _variables(
                                            finding
                                        ).get(
                                            "time"
                                        ),

                                    "dimension_column":
                                        _variables(
                                            finding
                                        ).get(
                                            "dimension"
                                        ),

                                    "entity_column":
                                        _variables(
                                            finding
                                        ).get(
                                            "entity"
                                        ),

                                    "aggregation_function":
                                        "",

                                    "ranking_order":
                                        "",

                                    "ranking_limit":
                                        None,

                                    "window_operation":
                                        "",

                                    "window_size":
                                        None,

                                    "blockers":
                                        [],

                                    "reasons":
                                        reasons,

                                    "confidence":
                                        1.0,
                                },

                            "contract":
                                contract,

                            "errors":
                                [],

                            "warnings":
                                caveats,

                            "normalizations":
                                [],
                        }
                    ],

                "attempt_count":
                    1,

                "retry_count":
                    0,

                "retry_triggered":
                    False,

                "retry_feedback":
                    [],

                "normalization_count":
                    0,

                "normalization_applied":
                    False,

                "planner_rule_version":
                    UNIFIED_REPORT_ARTIFACT_RULE_VERSION,
            },

        "validated_contract_count":
            1,

        "pipeline_item_count":
            1,

        "executed_count":
            1,

        "not_supported_count":
            0,

        "rejected_count":
            0,

        "items":
            [
                {
                    "contract_id":
                        contract_id,

                    "family":
                        family,

                    "pipeline_status":
                        "executed",

                    "native_tool":
                        {
                            "model":
                                "python-deterministic",

                            "contract_family":
                                family,

                            "available_tools":
                                [
                                    _requested_tool_name(
                                        family
                                    )
                                ],

                            "expected_tool":
                                _requested_tool_name(
                                    family
                                ),

                            "tool_call_received":
                                True,

                            "requested_tool":
                                _requested_tool_name(
                                    family
                                ),

                            "requested_arguments":
                                {
                                    "family":
                                        family,

                                    "dataset_ids":
                                        datasets,

                                    "analytical_grain":
                                        finding.get(
                                            "analytical_grain"
                                        ),

                                    "variables":
                                        _variables(
                                            finding
                                        ),
                                },

                            "validation_status":
                                "validated",

                            "validation_errors":
                                [],

                            "attempt_count":
                                1,

                            "retry_count":
                                0,

                            "attempts":
                                [],

                            "execution":
                                {
                                    "tool_name":
                                        _requested_tool_name(
                                            family
                                        ),

                                    "execution_status":
                                        "executed",

                                    "dataset_id":
                                        (
                                            str(
                                                dataset_id
                                            )
                                            if dataset_id
                                            is not None
                                            else
                                            None
                                        ),

                                    "dataset_filename":
                                        (
                                            str(
                                                finding.get(
                                                    "dataset_filename"
                                                )
                                            )
                                            if finding.get(
                                                "dataset_filename"
                                            )
                                            is not None
                                            else
                                            None
                                        ),

                                    "arguments":
                                        {
                                            "family":
                                                family,

                                            "dataset_ids":
                                                datasets,

                                            "analytical_grain":
                                                finding.get(
                                                    "analytical_grain"
                                                ),

                                            "variables":
                                                _variables(
                                                    finding
                                                ),
                                        },

                                    "result":
                                        result,

                                    "errors":
                                        [],

                                    "warnings":
                                        caveats,
                                },

                            "native_tool_rule_version":
                                UNIFIED_REPORT_ARTIFACT_RULE_VERSION,
                        },

                    "errors":
                        [],

                    "warnings":
                        caveats,
                }
            ],

        "notes":
            [
                (
                    "Server-owned report artifact adapted from "
                    "a deterministic UnifiedAnalysisReport finding."
                )
            ],

        "pipeline_rule_version":
            UNIFIED_REPORT_ARTIFACT_RULE_VERSION,

        "requested_finding":
            (
                dict(
                    finding
                )
                if (
                    source_type
                    in
                    REQUESTED_ANALYSIS_SOURCE_TYPES
                )
                else
                None
            ),

        "report_context":
            {
                "source_analysis_id":
                    source_analysis_id,

                "source_type":
                    source_type,

                "source_filename":
                    finding.get(
                        "source_filename"
                    ),

                "source_locator":
                    finding.get(
                        "source_locator"
                    ),

                "page_number":
                    finding.get(
                        "page_number"
                    ),

                "evidence_quote":
                    finding.get(
                        "evidence_quote"
                    ),

                "origin":
                    finding.get(
                        "origin"
                    ),

                "adapter_rule_version":
                    finding.get(
                        "adapter_rule_version"
                    ),
            },
    }


def _build_finding_record(
    *,
    workflow_id: str,
    source_type: AnalysisSourceType,
    finding: Any,
    select_by_default: bool,
    requested_plan: Any | None = None,
) -> AnalysisArtifactRecord:
    # Compatibility parameter only. Artifact registration and
    # report composition are intentionally separate concerns.
    _ = select_by_default


    payload = (
        _dump(
            finding
        )
    )


    source_analysis_id = str(
        payload.get(
            "analysis_id",
            "",
        )
        or
        payload.get(
            "request_id",
            "",
        )
        or
        payload.get(
            "title",
            "",
        )
    ).strip()


    if not (
        source_analysis_id
    ):
        raise ValueError(
            (
                "A report finding must expose analysis_id, "
                "request_id or title."
            )
        )


    objective = str(
        payload.get(
            "request_text",
            "",
        )
        or
        payload.get(
            "title",
            "",
        )
        or
        source_analysis_id
    ).strip()


    artifact_id = (
        _stable_id(
            workflow_id=
                workflow_id,

            source_type=
                source_type,

            source_analysis_id=
                source_analysis_id,
        )
    )


    trace_id = (
        f"report:{source_analysis_id}"
    )


    pipeline_payload = (
        _synthetic_native_payload(
            artifact_id=
                artifact_id,

            source_type=
                source_type,

            objective=
                objective,

            finding=
                payload,
        )
    )


    if (
        source_type
        in
        REQUESTED_ANALYSIS_SOURCE_TYPES
        and
        requested_plan
        is not None
    ):
        pipeline_payload[
            "requested_plan"
        ] = (
            _dump(
                requested_plan
            )
        )


    return (
        build_server_owned_analysis_record(
            workflow_id=
                workflow_id,

            analysis_id=
                artifact_id,

            trace_id=
                trace_id,

            source_type=
                source_type,

            objective=
                objective,

            executed=
                True,

            executed_count=
                1,

            pipeline_payload=
                pipeline_payload,

            # Report composition is manual-only.
            #
            # Keep the compatibility argument in this helper so
            # older server call sites do not break, but never
            # translate artifact registration into a report
            # selection side effect.
            select_by_default=
                False,
        )
    )




def _register_finding(
    *,
    workflow_id: str,
    source_type: AnalysisSourceType,
    finding: Any,
    select_by_default: bool,
    requested_plan: Any | None = None,
) -> AnalysisArtifactRecord:
    # Compatibility parameter only. Artifact registration and
    # report composition are intentionally separate concerns.
    _ = select_by_default


    payload = (
        _dump(
            finding
        )
    )


    source_analysis_id = str(
        payload.get(
            "analysis_id",
            "",
        )
        or
        payload.get(
            "request_id",
            "",
        )
        or
        payload.get(
            "title",
            "",
        )
    ).strip()


    if not (
        source_analysis_id
    ):
        raise ValueError(
            (
                "A report finding must expose analysis_id, "
                "request_id or title."
            )
        )


    objective = str(
        payload.get(
            "request_text",
            "",
        )
        or
        payload.get(
            "title",
            "",
        )
        or
        source_analysis_id
    ).strip()


    artifact_id = (
        _stable_id(
            workflow_id=
                workflow_id,

            source_type=
                source_type,

            source_analysis_id=
                source_analysis_id,
        )
    )


    trace_id = (
        f"report:{source_analysis_id}"
    )


    pipeline_payload = (
        _synthetic_native_payload(
            artifact_id=
                artifact_id,

            source_type=
                source_type,

            objective=
                objective,

            finding=
                payload,
        )
    )


    if (
        source_type
        in
        REQUESTED_ANALYSIS_SOURCE_TYPES
        and
        requested_plan
        is not None
    ):
        pipeline_payload[
            "requested_plan"
        ] = (
            _dump(
                requested_plan
            )
        )


    return (
        register_server_owned_analysis(
            workflow_id=
                workflow_id,

            analysis_id=
                artifact_id,

            trace_id=
                trace_id,

            source_type=
                source_type,

            objective=
                objective,

            executed=
                True,

            executed_count=
                1,

            pipeline_payload=
                pipeline_payload,

            # Report composition is manual-only.
            #
            # Keep the compatibility argument in this helper so
            # older server call sites do not break, but never
            # translate artifact registration into a report
            # selection side effect.
            select_by_default=
                False,
        )
    )






# ============================================================
# REQUEST LIFECYCLE ARTIFACTS
# ============================================================

def build_unresolved_requested_analysis_artifacts(
    *,
    workflow_id: str,
    execution_report: Any,
    plan_report: Any,
    source_type: AnalysisSourceType = "document_request",
) -> list[
    AnalysisArtifactRecord
]:
    """
    Persist documentary analytical requests that did not
    produce a reportable analytical result.

    These records are lifecycle artifacts, not findings.

    Important invariants:

    - they retain the same server-owned identity that a future
      successful requested finding will use;
    - executed=False keeps them outside report selection;
    - planner status and blockers remain server-owned;
    - no blocked or ambiguous request is converted into an
      observed analytical result.
    """

    source_type = (
        _require_requested_analysis_source_type(
            source_type
        )
    )


    from app.reporting.requested_adapter import (
        REPORTABLE_REQUESTED_STATUSES,
        build_request_plan_map,
        requested_analysis_id,
    )


    plan_map = (
        build_request_plan_map(
            plan_report
        )
    )


    registered: list[
        AnalysisArtifactRecord
    ] = []


    seen_source_ids: set[
        str
    ] = set()


    results = (
        getattr(
            execution_report,
            "results",
            [],
        )
        or
        []
    )


    for (
        request_order,
        execution,
    ) in enumerate(
        results,
        start=1,
    ):
        if (
            execution.execution_status
            in
            REPORTABLE_REQUESTED_STATUSES
        ):
            continue


        plan = (
            plan_map.get(
                execution.request_id
            )
        )


        if plan is None:
            raise ValueError(
                (
                    "No Request Planner record was found "
                    "for unresolved requested execution "
                    f"{execution.request_id}."
                )
            )


        if (
            execution.request_id
            !=
            plan.request_id
        ):
            raise ValueError(
                (
                    "Requested lifecycle execution / plan "
                    "request_id mismatch: "
                    f"{execution.request_id} != "
                    f"{plan.request_id}"
                )
            )


        source_analysis_id = (
            requested_analysis_id(
                execution
            )
        )


        if (
            source_analysis_id
            in
            seen_source_ids
        ):
            raise ValueError(
                (
                    "Duplicate unresolved requested "
                    "analysis identity detected: "
                    f"{source_analysis_id}"
                )
            )


        seen_source_ids.add(
            source_analysis_id
        )


        artifact_id = (
            _stable_id(
                workflow_id=
                    workflow_id,

                source_type=
                    source_type,

                source_analysis_id=
                    source_analysis_id,
            )
        )


        objective = str(
            execution.request_text
            or
            source_analysis_id
        ).strip()


        execution_payload = (
            _dump(
                execution
            )
        )


        plan_payload = (
            _dump(
                plan
            )
        )


        lifecycle_payload = {
            "artifact_kind":
                "requested_analysis_lifecycle",

            "status":
                execution.execution_status,

            "request_lifecycle":
                {
                    "request_id":
                        execution.request_id,

                    "request_text":
                        execution.request_text,

                    "request_order":
                        request_order,

                    "plan_status":
                        execution.plan_status,

                    "execution_status":
                        execution.execution_status,

                    "inferential_status":
                        execution.inferential_status,

                    "warnings":
                        list(
                            execution.warnings
                            or
                            []
                        ),

                    "limitations":
                        list(
                            execution.limitations
                            or
                            []
                        ),

                    "source_filename":
                        plan.source_filename,

                    "source_locator":
                        plan.source_locator,

                    "page_number":
                        plan.page_number,

                    "source_chunk_id":
                        plan.source_chunk_id,

                    "evidence_unit_id":
                        plan.evidence_unit_id,

                    "evidence_quote":
                        plan.evidence_quote,
                },

            "requested_execution":
                execution_payload,

            "requested_plan":
                plan_payload,
        }


        registered.append(
            build_server_owned_analysis_record(
                workflow_id=
                    workflow_id,

                analysis_id=
                    artifact_id,

                trace_id=
                    (
                        "report:"
                        +
                        source_analysis_id
                    ),

                source_type=
                    source_type,

                objective=
                    objective,

                executed=
                    False,

                executed_count=
                    0,

                pipeline_payload=
                    lifecycle_payload,

                select_by_default=
                    False,
            )
        )


    return registered




def register_unresolved_requested_analysis_artifacts(
    *,
    workflow_id: str,
    execution_report: Any,
    plan_report: Any,
    source_type: AnalysisSourceType = "document_request",
) -> list[
    AnalysisArtifactRecord
]:
    """
    Persist documentary analytical requests that did not
    produce a reportable analytical result.

    These records are lifecycle artifacts, not findings.

    Important invariants:

    - they retain the same server-owned identity that a future
      successful requested finding will use;
    - executed=False keeps them outside report selection;
    - planner status and blockers remain server-owned;
    - no blocked or ambiguous request is converted into an
      observed analytical result.
    """

    source_type = (
        _require_requested_analysis_source_type(
            source_type
        )
    )


    from app.reporting.requested_adapter import (
        REPORTABLE_REQUESTED_STATUSES,
        build_request_plan_map,
        requested_analysis_id,
    )


    plan_map = (
        build_request_plan_map(
            plan_report
        )
    )


    registered: list[
        AnalysisArtifactRecord
    ] = []


    seen_source_ids: set[
        str
    ] = set()


    results = (
        getattr(
            execution_report,
            "results",
            [],
        )
        or
        []
    )


    for (
        request_order,
        execution,
    ) in enumerate(
        results,
        start=1,
    ):
        if (
            execution.execution_status
            in
            REPORTABLE_REQUESTED_STATUSES
        ):
            continue


        plan = (
            plan_map.get(
                execution.request_id
            )
        )


        if plan is None:
            raise ValueError(
                (
                    "No Request Planner record was found "
                    "for unresolved requested execution "
                    f"{execution.request_id}."
                )
            )


        if (
            execution.request_id
            !=
            plan.request_id
        ):
            raise ValueError(
                (
                    "Requested lifecycle execution / plan "
                    "request_id mismatch: "
                    f"{execution.request_id} != "
                    f"{plan.request_id}"
                )
            )


        source_analysis_id = (
            requested_analysis_id(
                execution
            )
        )


        if (
            source_analysis_id
            in
            seen_source_ids
        ):
            raise ValueError(
                (
                    "Duplicate unresolved requested "
                    "analysis identity detected: "
                    f"{source_analysis_id}"
                )
            )


        seen_source_ids.add(
            source_analysis_id
        )


        artifact_id = (
            _stable_id(
                workflow_id=
                    workflow_id,

                source_type=
                    source_type,

                source_analysis_id=
                    source_analysis_id,
            )
        )


        objective = str(
            execution.request_text
            or
            source_analysis_id
        ).strip()


        execution_payload = (
            _dump(
                execution
            )
        )


        plan_payload = (
            _dump(
                plan
            )
        )


        lifecycle_payload = {
            "artifact_kind":
                "requested_analysis_lifecycle",

            "status":
                execution.execution_status,

            "request_lifecycle":
                {
                    "request_id":
                        execution.request_id,

                    "request_text":
                        execution.request_text,

                    "request_order":
                        request_order,

                    "plan_status":
                        execution.plan_status,

                    "execution_status":
                        execution.execution_status,

                    "inferential_status":
                        execution.inferential_status,

                    "warnings":
                        list(
                            execution.warnings
                            or
                            []
                        ),

                    "limitations":
                        list(
                            execution.limitations
                            or
                            []
                        ),

                    "source_filename":
                        plan.source_filename,

                    "source_locator":
                        plan.source_locator,

                    "page_number":
                        plan.page_number,

                    "source_chunk_id":
                        plan.source_chunk_id,

                    "evidence_unit_id":
                        plan.evidence_unit_id,

                    "evidence_quote":
                        plan.evidence_quote,
                },

            "requested_execution":
                execution_payload,

            "requested_plan":
                plan_payload,
        }


        registered.append(
            register_server_owned_analysis(
                workflow_id=
                    workflow_id,

                analysis_id=
                    artifact_id,

                trace_id=
                    (
                        "report:"
                        +
                        source_analysis_id
                    ),

                source_type=
                    source_type,

                objective=
                    objective,

                executed=
                    False,

                executed_count=
                    0,

                pipeline_payload=
                    lifecycle_payload,

                select_by_default=
                    False,
            )
        )


    return registered






# ============================================================
# PUBLIC SYNC
# ============================================================

def register_requested_report_finding(
    *,
    workflow_id: str,
    finding: Any,
    requested_plan: Any | None = None,
    expected_analysis_id: str | None = None,
    select_by_default: bool = False,
    source_type: AnalysisSourceType = "document_request",
) -> AnalysisArtifactRecord:
    """
    Persist one reportable documentary request finding.

    This helper intentionally reuses the same deterministic
    artifact identity as register_unified_report_artifacts().

    expected_analysis_id is an integrity guard used when an
    unresolved lifecycle artifact is promoted from
    executed=False to executed=True.

    The function validates the future artifact identity before
    performing the write.

    requested_plan, when supplied, must be the server-owned
    resolved RequestedAnalysisPlan used to produce the finding.
    It is persisted for later controlled reconfiguration; the
    browser never supplies this plan.

    Report selection is manual-only. ``select_by_default`` is
    retained for compatibility with older server call sites but
    is intentionally ignored when this finding is registered.
    """
    source_type = (
        _require_requested_analysis_source_type(
            source_type
        )
    )


    # Compatibility parameter only. A server caller may still
    # pass True, but report composition remains an explicit
    # user action.
    _ = select_by_default


    payload = (
        _dump(
            finding
        )
    )


    source_analysis_id = str(
        payload.get(
            "analysis_id",
            "",
        )
    ).strip()


    if not (
        source_analysis_id
    ):
        raise ValueError(
            (
                "A requested report finding must expose "
                "a stable analysis_id."
            )
        )


    artifact_id = (
        _stable_id(
            workflow_id=
                workflow_id,

            source_type=
                source_type,

            source_analysis_id=
                source_analysis_id,
        )
    )


    if (
        expected_analysis_id
        is not None
        and
        artifact_id
        !=
        str(
            expected_analysis_id
        ).strip()
    ):
        raise ValueError(
            (
                "Requested report finding identity does not "
                "match the existing lifecycle artifact. "
                f"expected={expected_analysis_id}, "
                f"computed={artifact_id}"
            )
        )


    return (
        _register_finding(
            workflow_id=
                workflow_id,

            source_type=
                source_type,

            finding=
                finding,

            select_by_default=
                False,

            requested_plan=
                requested_plan,
        )
    )


def build_unified_report_artifacts(
    *,
    workflow_id: str,
    report: Any,
) -> list[
    AnalysisArtifactRecord
]:
    """
    Persist reportable deterministic analyses produced by the
    standard DataLens analysis pipeline.

    Report-selection policy:

    - document_request -> available, not selected;
    - automatic        -> available, not selected.

    Prompt-native analyses are registered separately by the
    AI-native pipeline and are also available without being
    selected automatically.

    Executing or persisting an analysis must never mutate report
    composition. Selection is an explicit user action.
    """

    report_payload = (
        _dump(
            report
        )
    )


    registered: list[
        AnalysisArtifactRecord
    ] = []


    seen_source_keys: set[
        tuple[
            AnalysisSourceType,
            str,
        ]
    ] = set()


    requested = (
        report_payload.get(
            "requested_findings",
            [],
        )
    )


    if isinstance(
        requested,
        list,
    ):
        for finding in (
            requested
        ):
            if not isinstance(
                finding,
                dict,
            ):
                continue


            source_key = str(
                finding.get(
                    "analysis_id",
                    "",
                )
                or
                finding.get(
                    "request_id",
                    "",
                )
            ).strip()


            dedupe_key = (
                "document_request",
                source_key,
            )


            if (
                source_key
                and
                dedupe_key
                in seen_source_keys
            ):
                continue


            if (
                source_key
            ):
                seen_source_keys.add(
                    dedupe_key
                )


            registered.append(
                _build_finding_record(
                    workflow_id=
                        workflow_id,

                    source_type=
                        "document_request",

                    finding=
                        finding,

                    select_by_default=
                        False,
                )
            )


    for field_name in (
        "main_findings",
        "additional_findings",
        "context_analyses",
    ):
        findings = (
            report_payload.get(
                field_name,
                [],
            )
        )


        if not isinstance(
            findings,
            list,
        ):
            continue


        for finding in (
            findings
        ):
            if not isinstance(
                finding,
                dict,
            ):
                continue


            source_key = str(
                finding.get(
                    "analysis_id",
                    "",
                )
                or
                finding.get(
                    "title",
                    "",
                )
            ).strip()


            dedupe_key = (
                "automatic",
                source_key,
            )


            if (
                source_key
                and
                dedupe_key
                in seen_source_keys
            ):
                continue


            if (
                source_key
            ):
                seen_source_keys.add(
                    dedupe_key
                )


            registered.append(
                _build_finding_record(
                    workflow_id=
                        workflow_id,

                    source_type=
                        "automatic",

                    finding=
                        finding,

                    select_by_default=
                        False,
                )
            )


    return registered




def register_unified_report_artifacts(
    *,
    workflow_id: str,
    report: Any,
) -> list[
    AnalysisArtifactRecord
]:
    """
    Persist reportable deterministic analyses produced by the
    standard DataLens analysis pipeline.

    Report-selection policy:

    - document_request -> available, not selected;
    - automatic        -> available, not selected.

    Prompt-native analyses are registered separately by the
    AI-native pipeline and are also available without being
    selected automatically.

    Executing or persisting an analysis must never mutate report
    composition. Selection is an explicit user action.
    """

    report_payload = (
        _dump(
            report
        )
    )


    registered: list[
        AnalysisArtifactRecord
    ] = []


    seen_source_keys: set[
        tuple[
            AnalysisSourceType,
            str,
        ]
    ] = set()


    requested = (
        report_payload.get(
            "requested_findings",
            [],
        )
    )


    if isinstance(
        requested,
        list,
    ):
        for finding in (
            requested
        ):
            if not isinstance(
                finding,
                dict,
            ):
                continue


            source_key = str(
                finding.get(
                    "analysis_id",
                    "",
                )
                or
                finding.get(
                    "request_id",
                    "",
                )
            ).strip()


            dedupe_key = (
                "document_request",
                source_key,
            )


            if (
                source_key
                and
                dedupe_key
                in seen_source_keys
            ):
                continue


            if (
                source_key
            ):
                seen_source_keys.add(
                    dedupe_key
                )


            registered.append(
                _register_finding(
                    workflow_id=
                        workflow_id,

                    source_type=
                        "document_request",

                    finding=
                        finding,

                    select_by_default=
                        False,
                )
            )


    for field_name in (
        "main_findings",
        "additional_findings",
        "context_analyses",
    ):
        findings = (
            report_payload.get(
                field_name,
                [],
            )
        )


        if not isinstance(
            findings,
            list,
        ):
            continue


        for finding in (
            findings
        ):
            if not isinstance(
                finding,
                dict,
            ):
                continue


            source_key = str(
                finding.get(
                    "analysis_id",
                    "",
                )
                or
                finding.get(
                    "title",
                    "",
                )
            ).strip()


            dedupe_key = (
                "automatic",
                source_key,
            )


            if (
                source_key
                and
                dedupe_key
                in seen_source_keys
            ):
                continue


            if (
                source_key
            ):
                seen_source_keys.add(
                    dedupe_key
                )


            registered.append(
                _register_finding(
                    workflow_id=
                        workflow_id,

                    source_type=
                        "automatic",

                    finding=
                        finding,

                    select_by_default=
                        False,
                )
            )


    return registered






def register_contextualized_report_artifacts_atomic(
    *,
    workflow_id: str,
    execution_report: Any,
    plan_report: Any,
    report: Any,
) -> list[
    AnalysisArtifactRecord
]:
    """
    Commit the complete contextualized-report artifact set as
    one logical AnalysisArtifact transaction.

    No unresolved lifecycle artifact or reportable finding is
    made visible unless the complete metadata batch commits.
    """

    unresolved_records = (
        build_unresolved_requested_analysis_artifacts(
            workflow_id=
                workflow_id,

            execution_report=
                execution_report,

            plan_report=
                plan_report,
        )
    )


    report_records = (
        build_unified_report_artifacts(
            workflow_id=
                workflow_id,

            report=
                report,
        )
    )


    records = [
        *unresolved_records,
        *report_records,
    ]


    return (
        persist_server_owned_analysis_records_atomic(
            records=
                records
        )
    )
