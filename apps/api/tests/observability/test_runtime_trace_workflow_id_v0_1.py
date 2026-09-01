from __future__ import annotations

import inspect
import unittest

from app.api.analysis_run import (
    run_ai_native_pipeline,
)

from app.observability.runtime_trace import (
    RUNTIME_TRACE_RULE_VERSION,
    RUNTIME_WORKFLOW_ID_STATE_KEY,
    _workflow_id_from_scope,
    build_runtime_trace,
    stamp_validated_runtime_workflow_id,
)


VALID_WORKFLOW_ID = 'prep:b65b6d712ec34af29998feac0fd094a3'


class RuntimeTraceWorkflowIdV01Tests(
    unittest.TestCase
):
    def test_runtime_trace_version_remains_v01(
        self,
    ) -> None:
        self.assertEqual(
            RUNTIME_TRACE_RULE_VERSION,
            "runtime_trace_v0.1",
        )


    def test_validated_workflow_can_be_stamped(
        self,
    ) -> None:
        scope = {
            "type":
                "http",

            "method":
                "POST",

            "state":
                {},
        }


        result = (
            stamp_validated_runtime_workflow_id(
                scope=
                    scope,

                workflow_id=
                    VALID_WORKFLOW_ID,
            )
        )


        self.assertEqual(
            result,
            VALID_WORKFLOW_ID,
        )


        self.assertEqual(
            scope[
                "state"
            ][
                RUNTIME_WORKFLOW_ID_STATE_KEY
            ],
            VALID_WORKFLOW_ID,
        )


        self.assertEqual(
            _workflow_id_from_scope(
                scope
            ),
            VALID_WORKFLOW_ID,
        )


        trace = build_runtime_trace(
            request_id=
                (
                    "http:"
                    "00000000000000000000000000000000"
                ),

            scope=
                scope,

            status_code=
                200,

            duration_ms=
                10.0,
        )


        self.assertEqual(
            trace.workflow_id,
            VALID_WORKFLOW_ID,
        )


    def test_invalid_workflow_cannot_be_stamped(
        self,
    ) -> None:
        scope = {
            "type":
                "http",

            "method":
                "POST",

            "state":
                {},
        }


        with self.assertRaises(
            ValueError
        ):
            stamp_validated_runtime_workflow_id(
                scope=
                    scope,

                workflow_id=
                    "../../spoof",
            )


        self.assertNotIn(
            RUNTIME_WORKFLOW_ID_STATE_KEY,
            scope[
                "state"
            ],
        )


    def test_runtime_resolver_ignores_untrusted_transport(
        self,
    ) -> None:
        scope = {
            "type":
                "http",

            "method":
                "POST",

            "state":
                {},

            "path_params":
                {
                    "workflow_id":
                        VALID_WORKFLOW_ID,
                },

            "query_string":
                (
                    b"workflow_id="
                    +
                    VALID_WORKFLOW_ID.encode(
                        "utf-8"
                    )
                ),

            "headers":
                [
                    (
                        b"x-workflow-id",
                        VALID_WORKFLOW_ID.encode(
                            "utf-8"
                        ),
                    )
                ],
        }


        self.assertIsNone(
            _workflow_id_from_scope(
                scope
            )
        )


    def test_ai_native_request_is_direct_call_compatible(
        self,
    ) -> None:
        signature = inspect.signature(
            run_ai_native_pipeline
        )


        parameter = (
            signature.parameters[
                "request"
            ]
        )


        self.assertIsNone(
            parameter.default
        )


    def test_publication_occurs_after_authoritative_handoff(
        self,
    ) -> None:
        source = inspect.getsource(
            run_ai_native_pipeline
        )


        handoff_index = source.index(
            "load_validated_analysis_input_for_http("
        )


        guard_index = source.index(
            "if request is not None:"
        )


        stamp_index = source.index(
            "stamp_validated_runtime_workflow_id("
        )


        self.assertLess(
            handoff_index,
            guard_index,
        )


        self.assertLess(
            guard_index,
            stamp_index,
        )


    def test_publication_uses_request_scope(
        self,
    ) -> None:
        source = inspect.getsource(
            run_ai_native_pipeline
        )


        self.assertIn(
            "request.scope",
            source,
        )


if __name__ == "__main__":
    print(
        "=== DATALENS RUNTIME TRACE WORKFLOW CORRELATION v0.1 ==="
    )

    unittest.main(
        verbosity=2
    )
