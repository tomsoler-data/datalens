from __future__ import annotations


from collections.abc import (
    Sequence,
)

from typing import (
    Any,
)


SEMANTIC_VALUE_SAMPLE_BOUNDARY_RULE_VERSION = (
    "semantic_value_sample_boundary_v0.1"
)


# A model may receive a very small number of explicit
# dataset-derived values when semantic interpretation cannot
# be performed from metadata alone.
#
# This is intentionally not a raw-row allowance.
MAX_SEMANTIC_VALUE_SAMPLE_VALUES = 5


class SemanticValueSampleBoundaryError(
    ValueError
):
    """
    Raised before model transport when an explicit semantic
    value sample exceeds the DataLens privacy budget.
    """


def require_bounded_semantic_value_sample(
    values: Sequence[
        Any
    ],
    *,
    field_name: str,
) -> None:

    if isinstance(
        values,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise (
            SemanticValueSampleBoundaryError(
                (
                    "Semantic value sample field "
                    f"'{field_name}' must be a sequence "
                    "of individual values."
                )
            )
        )


    if not isinstance(
        values,
        Sequence,
    ):
        raise (
            SemanticValueSampleBoundaryError(
                (
                    "Semantic value sample field "
                    f"'{field_name}' has an unsupported "
                    "container type."
                )
            )
        )


    if (
        len(
            values
        )
        >
        MAX_SEMANTIC_VALUE_SAMPLE_VALUES
    ):
        raise (
            SemanticValueSampleBoundaryError(
                (
                    "Semantic value sample exceeds "
                    "the allowed privacy budget for "
                    f"'{field_name}'."
                )
            )
        )
