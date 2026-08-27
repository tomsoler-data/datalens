from __future__ import annotations

import os
import tempfile

from pathlib import Path


with tempfile.TemporaryDirectory() as directory:
    root = Path(
        directory
    )


    os.environ[
        "DATALENS_ANALYSIS_ARTIFACT_STORE_PATH"
    ] = str(
        root
        /
        "analysis_artifacts.json"
    )


    os.environ[
        "DATALENS_REPORT_SELECTION_STORE_PATH"
    ] = str(
        root
        /
        "report_selection.json"
    )


    from app.reporting.analysis_artifact_store import (
        register_server_owned_analysis,
    )

    from app.reporting.selected_report_pdf import (
        build_styles,
        build_unresolved_requested_section,
        unresolved_request_reason_lines,
        unresolved_request_status_label,
        unresolved_requested_analyses_for_workflow,
    )


    workflow_id = (
        "prep:test-pdf-unresolved"
    )


    def register_unresolved(
        *,
        suffix,
        request_text,
        request_order,
        plan_status,
        warning,
    ):
        register_server_owned_analysis(
            workflow_id=
                workflow_id,

            analysis_id=
                (
                    "analysis:report:"
                    +
                    suffix
                ),

            trace_id=
                (
                    "report:requested:"
                    +
                    suffix
                ),

            source_type=
                "document_request",

            objective=
                request_text,

            executed=
                False,

            executed_count=
                0,

            pipeline_payload=
                {
                    "artifact_kind":
                        "requested_analysis_lifecycle",

                    "status":
                        "not_executed",

                    "request_lifecycle":
                        {
                            "request_id":
                                (
                                    "request:"
                                    +
                                    suffix
                                ),

                            "request_text":
                                request_text,

                            "request_order":
                                request_order,

                            "plan_status":
                                plan_status,

                            "execution_status":
                                "not_executed",

                            "warnings":
                                [
                                    warning
                                ],

                            "limitations":
                                [],

                            "source_filename":
                                "Brief.pdf",

                            "source_locator":
                                "page 1",

                            "page_number":
                                1,

                            "evidence_quote":
                                request_text,
                        },
                },

            select_by_default=
                False,
        )


    # Deliberately register out of document order.
    register_unresolved(
        suffix=
            "btob",

        request_text=
            "repartition du CA BtoB",

        request_order=
            9,

        plan_status=
            "blocked",

        warning=
            "No defensible BtoB identification rule.",
    )


    register_unresolved(
        suffix=
            "top",

        request_text=
            "top produits",

        request_order=
            6,

        plan_status=
            "ambiguous",

        warning=
            "Ranking criterion is ambiguous.",
    )


    register_unresolved(
        suffix=
            "flop",

        request_text=
            "flop produits",

        request_order=
            7,

        plan_status=
            "ambiguous",

        warning=
            "Ranking criterion is ambiguous.",
    )


    # Executed documentary result must not enter this section.
    register_server_owned_analysis(
        workflow_id=
            workflow_id,

        analysis_id=
            "analysis:report:executed",

        trace_id=
            "report:requested:executed",

        source_type=
            "document_request",

        objective=
            "chiffre d'affaires",

        executed=
            True,

        executed_count=
            1,

        pipeline_payload=
            {
                "artifact_kind":
                    "requested_finding",
            },

        select_by_default=
            False,
    )


    requests = (
        unresolved_requested_analyses_for_workflow(
            workflow_id
        )
    )


    print()
    print(
        "===== PDF UNRESOLVED REQUESTS v0.1 ====="
    )
    print()


    assert (
        len(
            requests
        )
        ==
        3
    )


    print(
        "[PASS] exactly 3 unresolved requests loaded"
    )


    assert [
        item[
            "request_order"
        ]

        for item in requests
    ] == [
        6,
        7,
        9,
    ]


    print(
        "[PASS] original document order preserved"
    )


    assert [
        item[
            "request_text"
        ]

        for item in requests
    ] == [
        "top produits",
        "flop produits",
        "repartition du CA BtoB",
    ]


    assert (
        unresolved_request_status_label(
            requests[
                0
            ][
                "plan_status"
            ]
        )
        ==
        "Ambiguë"
    )


    assert (
        unresolved_request_status_label(
            requests[
                2
            ][
                "plan_status"
            ]
        )
        ==
        "Bloquée"
    )


    print(
        "[PASS] ambiguous / blocked presentation preserved"
    )


    for request in requests:
        reasons = (
            unresolved_request_reason_lines(
                request
            )
        )

        assert reasons

        assert all(
            isinstance(
                reason,
                str,
            )
            and
            reason.strip()

            for reason in reasons
        )


    print(
        "[PASS] server-owned blocker text remains available"
    )


    section = (
        build_unresolved_requested_section(
            workflow_id=
                workflow_id,

            styles=
                build_styles(),
        )
    )


    assert section


    print(
        "[PASS] PDF section flowables generated"
    )


    assert all(
        item[
            "execution_status"
        ]
        ==
        "not_executed"

        for item in requests
    )


    print(
        "[PASS] no unresolved request is represented "
        "as an executed result"
    )


    print()
    print(
        "Loaded:",
        len(
            requests
        ),
    )

    print(
        "Orders:",
        [
            item[
                "request_order"
            ]

            for item in requests
        ],
    )

    print(
        "Statuses:",
        [
            item[
                "plan_status"
            ]

            for item in requests
        ],
    )

    print()
    print(
        "PASS - selected PDF unresolved requests v0.1"
    )
