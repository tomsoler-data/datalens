from __future__ import annotations

from collections.abc import (
    Iterable,
)

from app.evaluation.decomposition_schemas import (
    CapabilityDecompositionReport,
    CapabilitySliceSummary,
)

from app.evaluation.runner_schemas import (
    SemanticBenchmarkSuiteResult,
)


# ============================================================
# VERSION
# ============================================================

DECOMPOSITION_RULE_VERSION = (
    "evaluation_capability_decomposition_v0.1"
)


# ============================================================
# DEFAULT SYSTEM-EXTENSION FIELDS
#
# These fields are produced by dedicated DataLens components
# and are not requested from the raw LLM semantic draft.
# ============================================================

DEFAULT_SYSTEM_EXTENSION_FIELDS = {
    "quantity_dimension",
    "quantity_unit",
}


# ============================================================
# HELPERS
# ============================================================

def _safe_accuracy(
    correct_count: int,
    assertion_count: int,
) -> float | None:
    if (
        assertion_count
        ==
        0
    ):
        return None


    return round(
        correct_count
        /
        assertion_count,
        6,
    )


def _safe_delta(
    datalens_accuracy: float | None,
    raw_accuracy: float | None,
) -> float | None:
    if (
        datalens_accuracy is None
        or
        raw_accuracy is None
    ):
        return None


    return round(
        datalens_accuracy
        -
        raw_accuracy,
        6,
    )


def _assertion_key(
    assertion,
) -> tuple[
    str,
    str,
    str,
]:
    return (
        assertion.case_id,
        assertion.subject,
        assertion.field,
    )


# ============================================================
# BUILD SLICE
# ============================================================

def _build_slice(
    *,
    label: str,
    keys: list[
        tuple[
            str,
            str,
            str,
        ]
    ],
    raw_index: dict,
    datalens_index: dict,
) -> CapabilitySliceSummary:
    raw_correct_count = sum(
        1

        for key
        in keys

        if raw_index[
            key
        ].correct
    )


    datalens_correct_count = sum(
        1

        for key
        in keys

        if datalens_index[
            key
        ].correct
    )


    assertion_count = len(
        keys
    )


    raw_accuracy = (
        _safe_accuracy(
            raw_correct_count,
            assertion_count,
        )
    )


    datalens_accuracy = (
        _safe_accuracy(
            datalens_correct_count,
            assertion_count,
        )
    )


    fields = sorted(
        {
            key[
                2
            ]

            for key
            in keys
        }
    )


    return CapabilitySliceSummary(
        label=
            label,

        fields=
            fields,

        assertion_count=
            assertion_count,

        raw_correct_count=
            raw_correct_count,

        datalens_correct_count=
            datalens_correct_count,

        raw_accuracy=
            raw_accuracy,

        datalens_accuracy=
            datalens_accuracy,

        correct_count_delta=(
            datalens_correct_count
            -
            raw_correct_count
        ),

        accuracy_delta=(
            _safe_delta(
                datalens_accuracy,
                raw_accuracy,
            )
        ),
    )


# ============================================================
# PUBLIC DECOMPOSITION
# ============================================================

def decompose_suite_capabilities(
    *,
    suite_result: SemanticBenchmarkSuiteResult,
    system_extension_fields: Iterable[
        str
    ] = DEFAULT_SYSTEM_EXTENSION_FIELDS,
) -> CapabilityDecompositionReport:
    extension_fields = {
        str(
            field
        )

        for field
        in system_extension_fields
    }


    raw_assertions = (
        suite_result.raw_columns.assertions
        +
        suite_result.raw_pairs.assertions
    )


    datalens_assertions = (
        suite_result.normalized_columns.assertions
        +
        suite_result.normalized_pairs.assertions
    )


    raw_index = {
        _assertion_key(
            assertion
        ):
            assertion

        for assertion
        in raw_assertions
    }


    datalens_index = {
        _assertion_key(
            assertion
        ):
            assertion

        for assertion
        in datalens_assertions
    }


    raw_keys = set(
        raw_index
    )


    datalens_keys = set(
        datalens_index
    )


    # ========================================================
    # FAIRNESS / CONSISTENCY CHECK
    #
    # The decomposition only makes sense when both versions
    # were evaluated against the exact same assertion set.
    # ========================================================

    if (
        raw_keys
        !=
        datalens_keys
    ):
        missing_from_raw = (
            datalens_keys
            -
            raw_keys
        )


        missing_from_datalens = (
            raw_keys
            -
            datalens_keys
        )


        raise ValueError(
            "Raw and DataLens assertion sets do not match. "
            f"Missing from raw: "
            f"{sorted(missing_from_raw)}. "
            f"Missing from DataLens: "
            f"{sorted(missing_from_datalens)}."
        )


    all_keys = sorted(
        datalens_keys
    )


    extension_keys = [
        key

        for key
        in all_keys

        if (
            key[
                2
            ]
            in extension_fields
        )
    ]


    shared_keys = [
        key

        for key
        in all_keys

        if (
            key[
                2
            ]
            not in extension_fields
        )
    ]


    shared = (
        _build_slice(
            label=
                "shared_capabilities",

            keys=
                shared_keys,

            raw_index=
                raw_index,

            datalens_index=
                datalens_index,
        )
    )


    system_extensions = (
        _build_slice(
            label=
                "system_extensions",

            keys=
                extension_keys,

            raw_index=
                raw_index,

            datalens_index=
                datalens_index,
        )
    )


    end_to_end = (
        _build_slice(
            label=
                "end_to_end",

            keys=
                all_keys,

            raw_index=
                raw_index,

            datalens_index=
                datalens_index,
        )
    )


    total_gain_count = (
        end_to_end.correct_count_delta
    )


    shared_gain_count = (
        shared.correct_count_delta
    )


    extension_gain_count = (
        system_extensions.correct_count_delta
    )


    # ========================================================
    # INTERNAL CONSISTENCY
    # ========================================================

    if (
        shared.assertion_count
        +
        system_extensions.assertion_count
        !=
        end_to_end.assertion_count
    ):
        raise ValueError(
            "Capability decomposition assertion counts "
            "do not reconcile."
        )


    if (
        shared_gain_count
        +
        extension_gain_count
        !=
        total_gain_count
    ):
        raise ValueError(
            "Capability decomposition gain counts "
            "do not reconcile."
        )


    # ========================================================
    # GAIN SHARES
    #
    # These are descriptive arithmetic shares.
    # They are NOT causal attribution estimates.
    # ========================================================

    if (
        total_gain_count
        >
        0
    ):
        shared_gain_share = round(
            shared_gain_count
            /
            total_gain_count,
            6,
        )


        extension_gain_share = round(
            extension_gain_count
            /
            total_gain_count,
            6,
        )

    else:
        shared_gain_share = None
        extension_gain_share = None


    return CapabilityDecompositionReport(
        benchmark_id=
            suite_result.benchmark_id,

        shared=
            shared,

        system_extensions=
            system_extensions,

        end_to_end=
            end_to_end,

        total_gain_count=
            total_gain_count,

        shared_gain_count=
            shared_gain_count,

        extension_gain_count=
            extension_gain_count,

        shared_gain_share=
            shared_gain_share,

        extension_gain_share=
            extension_gain_share,

        extension_fields=
            sorted(
                extension_fields
            ),
    )
