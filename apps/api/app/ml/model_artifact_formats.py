from __future__ import annotations


from typing import (
    Literal,
    cast,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_ARTIFACT_FORMAT_RULE_VERSION = (
    "ml_model_artifact_format_v0.1"
)


# ============================================================
# TYPES
# ============================================================


MLModelSerializationFormat = Literal[
    "joblib",
    "pytorch_bundle",
]


_FORMAT_SUFFIXES: dict[
    str,
    str,
] = {
    "joblib":
        ".joblib",

    "pytorch_bundle":
        ".ptbundle",
}


# ============================================================
# AUTHORITY
# ============================================================


def normalize_ml_model_serialization_format(
    value: object,
) -> MLModelSerializationFormat:

    normalized = str(
        value
        if value is not None
        else ""
    ).strip()


    if (
        normalized
        not in
        _FORMAT_SUFFIXES
    ):
        raise ValueError(
            (
                "Unsupported ML Model Artifact "
                "serialization format. "
                f"serialization_format={normalized!r}"
            )
        )


    return cast(
        MLModelSerializationFormat,
        normalized,
    )


def ml_model_serialization_suffix(
    serialization_format: object,
) -> str:

    normalized = (
        normalize_ml_model_serialization_format(
            serialization_format
        )
    )


    return (
        _FORMAT_SUFFIXES[
            normalized
        ]
    )
