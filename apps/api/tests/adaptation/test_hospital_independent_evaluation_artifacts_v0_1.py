from __future__ import annotations


import hashlib
import json
import tempfile


from pathlib import Path


from app.adaptation import (
    hospital_independent_evaluation_artifacts_v0_1
    as artifacts
)

from app.adaptation import (
    hospital_independent_evaluation_scoring_v0_1
    as scoring
)


EXECUTION_COMMIT = (
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)

CLAIMED_AT = (
    "2026-09-12T12:00:00Z"
)

COMPLETED_AT = (
    "2026-09-12T12:30:00Z"
)


def synthetic_gold(
) -> tuple[
    dict,
    ...,
]:

    records = []

    for relation in scoring.EXPECTED_RELATIONS:

        for index in range(
            scoring.EXPECTED_CASES_PER_RELATION
        ):

            records.append(
                {
                    "case_id":
                        (
                            f"{relation}"
                            f"::{index}"
                        ),

                    "gold_relation":
                        relation,
                }
            )

    return tuple(
        records
    )


def wrong_relation(
    relation: str,
) -> str:

    position = scoring.EXPECTED_RELATIONS.index(
        relation
    )

    return scoring.EXPECTED_RELATIONS[
        (
            position
            +
            1
        )
        %
        len(
            scoring.EXPECTED_RELATIONS
        )
    ]


def counts(
    *,
    default: int,
    **overrides: int,
) -> dict[
    str,
    int,
]:

    result = {
        relation:
            default

        for relation
        in scoring.EXPECTED_RELATIONS
    }

    result.update(
        overrides
    )

    return result


def synthetic_results(
    *,
    model_label: str,
    correct_counts: dict[
        str,
        int,
    ],
) -> tuple[
    dict,
    ...,
]:

    records = []

    for gold in synthetic_gold():

        case_id = gold[
            "case_id"
        ]

        relation = gold[
            "gold_relation"
        ]

        relation_index = int(
            case_id.rsplit(
                "::",
                1,
            )[
                1
            ]
        )

        if (
            relation_index
            <
            correct_counts[
                relation
            ]
        ):

            predicted = relation

        else:

            predicted = wrong_relation(
                relation
            )

        prompt_sha = hashlib.sha256(
            (
                "synthetic-prompt::"
                f"{case_id}"
            ).encode(
                "utf-8"
            )
        ).hexdigest()

        decoded_output = (
            '{"relation":"'
            f'{predicted}'
            '","reason":"'
            "Synthetic deterministic reason "
            "for artifact writer contract testing."
            '"}'
        )

        records.append(
            {
                "case_id":
                    case_id,

                "model_label":
                    model_label,

                "prompt_sha256":
                    prompt_sha,

                "strict_json_valid":
                    True,

                "predicted_relation":
                    predicted,

                "reason":
                    (
                        "Synthetic deterministic reason "
                        "for artifact writer contract testing."
                    ),

                "reason_word_count":
                    8,

                "invalid_reason":
                    None,

                "decoded_output":
                    decoded_output,

                "decoded_output_sha256":
                    hashlib.sha256(
                        decoded_output.encode(
                            "utf-8"
                        )
                    ).hexdigest(),

                "terminal_stop_token_id":
                    1,

                "generation_budget_exhausted":
                    False,

                "prompt_token_count":
                    100,

                "generated_token_count":
                    20,
            }
        )

    return tuple(
        records
    )


def base_results(
) -> tuple[
    dict,
    ...,
]:

    return synthetic_results(
        model_label=
            "base",

        correct_counts=
            counts(
                default=4
            ),
    )


def adapted_results(
) -> tuple[
    dict,
    ...,
]:

    adapted_counts = counts(
        default=4
    )

    adapted_counts[
        "same_metric_different_state"
    ] = 5

    return synthetic_results(
        model_label=
            "adapted",

        correct_counts=
            adapted_counts,
    )


def synthetic_scoring_result(
) -> dict:

    return scoring.score_official_comparison(
        gold_records=
            synthetic_gold(),

        base_results=
            base_results(),

        adapted_results=
            adapted_results(),
    )


def write_synthetic_marker(
    directory: Path,
) -> Path:

    path = (
        directory
        /
        artifacts.CONSUMPTION_MARKER_FILENAME
    )

    payload = {
        "rule_version":
            "synthetic-test-marker",

        "status":
            "claimed_before_holdout_open",

        "claimed_at_utc":
            CLAIMED_AT,

        "execution_authority_commit":
            EXECUTION_COMMIT,
    }

    path.write_bytes(
        artifacts.canonical_json_bytes(
            payload
        )
    )

    return path


def test_canonical_json_contract(
) -> None:

    encoded = artifacts.canonical_json_bytes(
        {
            "b": 2,
            "a": 1,
        }
    )

    assert encoded.endswith(
        b"\n"
    )

    assert b"\r" not in encoded

    assert encoded.startswith(
        b'{\n  "a": 1,'
    )

    assert (
        artifacts.sha256_bytes(
            encoded
        )
        ==
        hashlib.sha256(
            encoded
        ).hexdigest()
    )


def test_frozen_artifact_authorities(
) -> None:

    assert (
        artifacts.EXPECTED_PROTOCOL_SHA256
        ==
        (
            "0e958d67a6294f8666485a6274b1eee2"
            "1b9f06fe2e300ba93241a7ffaba3127d"
        )
    )

    assert (
        artifacts.EXPECTED_SCORER_SHA256
        ==
        (
            "9b9d648716f96f4a062ecb01e1e18b73"
            "a74aa0ff5e346a78048d0d7e77c8ab13"
        )
    )

    assert (
        artifacts.EXPECTED_SCORER_TEST_SHA256
        ==
        (
            "55851a2fe2a51e7f9e18fe60768fb868"
            "1286adc03955f5131a82b9690541cc4d"
        )
    )

    assert (
        artifacts.EXPECTED_CASE_COUNT
        ==
        30
    )


def test_predictions_boundary_matches_execution(
) -> None:

    payload = artifacts.build_predictions_artifact(
        execution_authority_commit=
            EXECUTION_COMMIT,

        claimed_at_utc=
            CLAIMED_AT,

        base_results=
            base_results(),

        adapted_results=
            adapted_results(),
    )

    assert payload[
        "case_count"
    ] == 30

    assert len(
        payload[
            "base_results"
        ]
    ) == 30

    assert len(
        payload[
            "adapted_results"
        ]
    ) == 30

    for label in (
        "base_results",
        "adapted_results",
    ):

        for record in payload[
            label
        ]:

            assert (
                "predicted_relation"
                in
                record
            )

            assert (
                "relation"
                not in
                record
            )

            assert (
                "gold_relation"
                not in
                record
            )

            assert (
                "gold_reason"
                not in
                record
            )


def test_predictions_reject_gold_material(
) -> None:

    base = list(
        base_results()
    )

    base[
        0
    ] = {
        **base[
            0
        ],

        "gold_reason":
            "Forbidden synthetic gold field.",
    }

    try:

        artifacts.build_predictions_artifact(
            execution_authority_commit=
                EXECUTION_COMMIT,

            claimed_at_utc=
                CLAIMED_AT,

            base_results=
                base,

            adapted_results=
                adapted_results(),
        )

    except RuntimeError:

        pass

    else:

        raise AssertionError(
            (
                "Gold material crossed predictions "
                "boundary."
            )
        )


def test_prompt_identity_mismatch_fails_closed(
) -> None:

    adapted = list(
        adapted_results()
    )

    adapted[
        0
    ] = {
        **adapted[
            0
        ],

        "prompt_sha256":
            hashlib.sha256(
                b"different-prompt"
            ).hexdigest(),
    }

    try:

        artifacts.build_predictions_artifact(
            execution_authority_commit=
                EXECUTION_COMMIT,

            claimed_at_utc=
                CLAIMED_AT,

            base_results=
                base_results(),

            adapted_results=
                adapted,
        )

    except RuntimeError:

        pass

    else:

        raise AssertionError(
            "Prompt identity mismatch was accepted."
        )


def test_report_preserves_deterministic_scorer(
) -> None:

    scoring_result = synthetic_scoring_result()

    assert (
        scoring_result[
            "promotion_eligible"
        ]
        is True
    )

    report = artifacts.build_report_artifact(
        execution_authority_commit=
            EXECUTION_COMMIT,

        claimed_at_utc=
            CLAIMED_AT,

        scoring_result=
            scoring_result,
    )

    assert (
        report[
            "scoring"
        ]
        ==
        scoring_result
    )

    assert (
        report[
            "promotion_eligible"
        ]
        is True
    )

    assert (
        report[
            "scorer_sha256"
        ]
        ==
        artifacts.EXPECTED_SCORER_SHA256
    )


def test_missing_marker_blocks_all_writes(
) -> None:

    with tempfile.TemporaryDirectory() as temporary:

        directory = Path(
            temporary
        )

        try:

            artifacts.write_official_artifact_chain(
                artifact_dir=
                    directory,

                execution_authority_commit=
                    EXECUTION_COMMIT,

                claimed_at_utc=
                    CLAIMED_AT,

                completed_at_utc=
                    COMPLETED_AT,

                base_results=
                    base_results(),

                adapted_results=
                    adapted_results(),

                scoring_result=
                    synthetic_scoring_result(),
            )

        except RuntimeError:

            pass

        else:

            raise AssertionError(
                (
                    "Writer accepted missing "
                    "consumption marker."
                )
            )

        paths = artifacts.single_use_artifact_paths(
            artifact_dir=
                directory
        )

        assert not paths[
            "predictions"
        ].exists()

        assert not paths[
            "report"
        ].exists()

        assert not paths[
            "receipt"
        ].exists()


def test_marker_identity_mismatch_blocks_writes(
) -> None:

    with tempfile.TemporaryDirectory() as temporary:

        directory = Path(
            temporary
        )

        marker = (
            directory
            /
            artifacts.CONSUMPTION_MARKER_FILENAME
        )

        marker.write_bytes(
            artifacts.canonical_json_bytes(
                {
                    "status":
                        "claimed_before_holdout_open",

                    "claimed_at_utc":
                        CLAIMED_AT,

                    "execution_authority_commit":
                        (
                            "bbbbbbbbbbbbbbbbbbbb"
                            "bbbbbbbbbbbbbbbbbbbb"
                        ),
                }
            )
        )

        try:

            artifacts.write_official_artifact_chain(
                artifact_dir=
                    directory,

                execution_authority_commit=
                    EXECUTION_COMMIT,

                claimed_at_utc=
                    CLAIMED_AT,

                completed_at_utc=
                    COMPLETED_AT,

                base_results=
                    base_results(),

                adapted_results=
                    adapted_results(),

                scoring_result=
                    synthetic_scoring_result(),
            )

        except RuntimeError:

            pass

        else:

            raise AssertionError(
                (
                    "Writer accepted mismatched "
                    "consumption marker."
                )
            )

        paths = artifacts.single_use_artifact_paths(
            artifact_dir=
                directory
        )

        assert not paths[
            "predictions"
        ].exists()

        assert not paths[
            "report"
        ].exists()

        assert not paths[
            "receipt"
        ].exists()


def test_exact_chain_and_receipt_bindings(
) -> None:

    with tempfile.TemporaryDirectory() as temporary:

        directory = Path(
            temporary
        )

        marker = write_synthetic_marker(
            directory
        )

        scoring_result = synthetic_scoring_result()

        result = artifacts.write_official_artifact_chain(
            artifact_dir=
                directory,

            execution_authority_commit=
                EXECUTION_COMMIT,

            claimed_at_utc=
                CLAIMED_AT,

            completed_at_utc=
                COMPLETED_AT,

            base_results=
                base_results(),

            adapted_results=
                adapted_results(),

            scoring_result=
                scoring_result,
        )

        paths = artifacts.single_use_artifact_paths(
            artifact_dir=
                directory
        )

        assert paths[
            "predictions"
        ].is_file()

        assert paths[
            "report"
        ].is_file()

        assert paths[
            "receipt"
        ].is_file()

        predictions = json.loads(
            paths[
                "predictions"
            ].read_text(
                encoding="utf-8"
            )
        )

        report = json.loads(
            paths[
                "report"
            ].read_text(
                encoding="utf-8"
            )
        )

        receipt = json.loads(
            paths[
                "receipt"
            ].read_text(
                encoding="utf-8"
            )
        )

        assert (
            receipt[
                "status"
            ]
            ==
            "completed"
        )

        assert (
            receipt[
                "execution_authority_commit"
            ]
            ==
            EXECUTION_COMMIT
        )

        assert (
            receipt[
                "consumption_marker_sha256"
            ]
            ==
            artifacts.sha256_file(
                marker
            )
        )

        assert (
            receipt[
                "predictions_sha256"
            ]
            ==
            artifacts.sha256_file(
                paths[
                    "predictions"
                ]
            )
        )

        assert (
            receipt[
                "report_sha256"
            ]
            ==
            artifacts.sha256_file(
                paths[
                    "report"
                ]
            )
        )

        assert (
            result[
                "receipt_sha256"
            ]
            ==
            artifacts.sha256_file(
                paths[
                    "receipt"
                ]
            )
        )

        assert (
            receipt[
                "promotion_eligible"
            ]
            ==
            report[
                "promotion_eligible"
            ]
        )

        assert (
            report[
                "scoring"
            ]
            ==
            scoring_result
        )

        assert (
            predictions[
                "base_results"
            ][
                0
            ][
                "case_id"
            ]
            ==
            synthetic_gold()[
                0
            ][
                "case_id"
            ]
        )

        assert (
            predictions[
                "base_results"
            ][
                0
            ][
                "prompt_sha256"
            ]
            ==
            predictions[
                "adapted_results"
            ][
                0
            ][
                "prompt_sha256"
            ]
        )


def test_second_write_fails_closed_without_replacement(
) -> None:

    with tempfile.TemporaryDirectory() as temporary:

        directory = Path(
            temporary
        )

        write_synthetic_marker(
            directory
        )

        kwargs = {
            "artifact_dir":
                directory,

            "execution_authority_commit":
                EXECUTION_COMMIT,

            "claimed_at_utc":
                CLAIMED_AT,

            "completed_at_utc":
                COMPLETED_AT,

            "base_results":
                base_results(),

            "adapted_results":
                adapted_results(),

            "scoring_result":
                synthetic_scoring_result(),
        }

        artifacts.write_official_artifact_chain(
            **kwargs
        )

        paths = artifacts.single_use_artifact_paths(
            artifact_dir=
                directory
        )

        before = {
            label:
                artifacts.sha256_file(
                    paths[
                        label
                    ]
                )

            for label in (
                "predictions",
                "report",
                "receipt",
            )
        }

        try:

            artifacts.write_official_artifact_chain(
                **kwargs
            )

        except RuntimeError:

            pass

        else:

            raise AssertionError(
                (
                    "Second official artifact "
                    "write was accepted."
                )
            )

        after = {
            label:
                artifacts.sha256_file(
                    paths[
                        label
                    ]
                )

            for label in (
                "predictions",
                "report",
                "receipt",
            )
        }

        assert before == after


def test_preexisting_output_blocks_before_new_write(
) -> None:

    with tempfile.TemporaryDirectory() as temporary:

        directory = Path(
            temporary
        )

        write_synthetic_marker(
            directory
        )

        paths = artifacts.single_use_artifact_paths(
            artifact_dir=
                directory
        )

        paths[
            "report"
        ].write_text(
            "{}\n",
            encoding="utf-8",
            newline="\n",
        )

        try:

            artifacts.write_official_artifact_chain(
                artifact_dir=
                    directory,

                execution_authority_commit=
                    EXECUTION_COMMIT,

                claimed_at_utc=
                    CLAIMED_AT,

                completed_at_utc=
                    COMPLETED_AT,

                base_results=
                    base_results(),

                adapted_results=
                    adapted_results(),

                scoring_result=
                    synthetic_scoring_result(),
            )

        except RuntimeError:

            pass

        else:

            raise AssertionError(
                (
                    "Preexisting output did not "
                    "block artifact chain."
                )
            )

        assert not paths[
            "predictions"
        ].exists()

        assert not paths[
            "receipt"
        ].exists()


def test_writer_has_no_model_or_protected_runtime_authority(
) -> None:

    source_path = (
        Path(
            artifacts.__file__
        )
        .resolve()
    )

    source = source_path.read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "load_frozen_holdout_for_execution",
        "HOLDOUT_CASES_PATH",
        "GOLD_PATH",
        "gold_reason_path",
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        ".generate(",
        ".from_pretrained(",
    ):

        assert forbidden not in source

    assert ".unlink(" not in source
    assert "os.remove(" not in source

    assert "os.O_EXCL" in source
    assert "os.fsync" in source


def test_real_evaluation_directory_remains_unconsumed(
) -> None:

    root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    artifact_dir = (
        root
        /
        "artifacts"
        /
        "adaptation"
        /
        "evaluation"
    )

    paths = artifacts.single_use_artifact_paths(
        artifact_dir=
            artifact_dir
    )

    assert not paths[
        "consumption_marker"
    ].exists()

    assert not paths[
        "predictions"
    ].exists()

    assert not paths[
        "report"
    ].exists()

    assert not paths[
        "receipt"
    ].exists()


TESTS = (
    test_canonical_json_contract,
    test_frozen_artifact_authorities,
    test_predictions_boundary_matches_execution,
    test_predictions_reject_gold_material,
    test_prompt_identity_mismatch_fails_closed,
    test_report_preserves_deterministic_scorer,
    test_missing_marker_blocks_all_writes,
    test_marker_identity_mismatch_blocks_writes,
    test_exact_chain_and_receipt_bindings,
    test_second_write_fails_closed_without_replacement,
    test_preexisting_output_blocks_before_new_write,
    test_writer_has_no_model_or_protected_runtime_authority,
    test_real_evaluation_directory_remains_unconsumed,
)


if __name__ == "__main__":

    print()
    print(
        "=== R8 HOSPITAL OFFICIAL ARTIFACT WRITER v0.1 ==="
    )
    print()

    for test in TESTS:

        test()

        print(
            f"[PASS] {test.__name__}"
        )

    print()
    print(
        "PASS - hospital official artifact writer v0.1"
    )
