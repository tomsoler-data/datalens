from __future__ import annotations


import math


from pydantic import (
    BaseModel,
    ConfigDict,
)


from app.model_lifecycle.fingerprints import (
    canonical_source_contract_json,
    source_contract_sha256,
)


class ExampleContract(
    BaseModel
):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    experiment_id: str
    seed: int
    nested: dict[
        str,
        object,
    ]


def expect_value_error(
    value: object,
) -> None:

    try:
        canonical_source_contract_json(
            value
        )

    except ValueError:
        return


    raise AssertionError(
        "Expected canonicalization failure."
    )


def main(
) -> None:

    left = {
        "experiment_id":
            "adaptation:test",

        "seed":
            42,

        "nested": {
            "rank": 16,
            "alpha": 32,
        },
    }


    right = {
        "nested": {
            "alpha": 32,
            "rank": 16,
        },

        "seed":
            42,

        "experiment_id":
            "adaptation:test",
    }


    left_json = (
        canonical_source_contract_json(
            left
        )
    )

    right_json = (
        canonical_source_contract_json(
            right
        )
    )


    assert (
        left_json
        ==
        right_json
    )


    left_sha = (
        source_contract_sha256(
            left
        )
    )

    right_sha = (
        source_contract_sha256(
            right
        )
    )


    assert (
        left_sha
        ==
        right_sha
    )

    assert len(
        left_sha
    ) == 64


    model = ExampleContract(
        **left
    )


    model_sha = (
        source_contract_sha256(
            model
        )
    )


    assert (
        model_sha
        ==
        left_sha
    )


    changed = {
        **left,
        "seed": 43,
    }


    changed_sha = (
        source_contract_sha256(
            changed
        )
    )


    assert (
        changed_sha
        !=
        left_sha
    )


    expect_value_error(
        [
            "not",
            "a",
            "contract",
        ]
    )


    expect_value_error(
        {
            "value":
                math.nan
        }
    )


    expect_value_error(
        {
            "value":
                math.inf
        }
    )


    expect_value_error(
        {
            "bytes":
                b"not-json"
        }
    )


    print(
        "Canonical key ordering                    PASS"
    )

    print(
        "Deterministic SHA-256                    PASS"
    )

    print(
        "Pydantic/mapping parity                 PASS"
    )

    print(
        "Contract mutation changes fingerprint   PASS"
    )

    print(
        "Non-contract root rejected              PASS"
    )

    print(
        "NaN / Infinity rejected                 PASS"
    )

    print(
        "Non-JSON bytes rejected                 PASS"
    )

    print()
    print(
        "Model Lifecycle Fingerprints v0.1: PASS"
    )


if __name__ == "__main__":
    main()
