from __future__ import annotations


import hashlib
import json


from collections.abc import Mapping


from pydantic import BaseModel


# ============================================================
# NORMALIZATION
# ============================================================


def _source_contract_payload(
    value: object,
) -> object:

    if isinstance(
        value,
        BaseModel,
    ):
        return value.model_dump(
            mode="json"
        )


    if isinstance(
        value,
        Mapping,
    ):
        return dict(
            value
        )


    raise ValueError(
        (
            "source contract must be a "
            "Pydantic model or JSON-style mapping."
        )
    )


# ============================================================
# CANONICAL JSON
# ============================================================


def canonical_source_contract_json(
    source_contract: object,
) -> str:
    """
    Produce deterministic JSON for one authoritative source
    contract without introducing model-family semantics.

    JSON object key order and whitespace cannot affect the
    resulting fingerprint. NaN and Infinity are rejected.
    """

    payload = (
        _source_contract_payload(
            source_contract
        )
    )


    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            allow_nan=False,
        )

    except (
        TypeError,
        ValueError,
    ) as error:

        raise ValueError(
            (
                "source contract could not be "
                "canonically serialized."
            )
        ) from error


# ============================================================
# SHA-256
# ============================================================


def source_contract_sha256(
    source_contract: object,
) -> str:

    canonical_json = (
        canonical_source_contract_json(
            source_contract
        )
    )


    return hashlib.sha256(
        canonical_json.encode(
            "utf-8"
        )
    ).hexdigest()
