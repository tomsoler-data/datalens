from __future__ import annotations

import inspect

from types import (
    SimpleNamespace,
)

from fastapi import (
    HTTPException,
)


from app.api.preparation_cleaning import (
    _require_complete_cleaning_decision,
    apply_uploaded_cleaning_plan,
)

from app.api import (
    preparation_combination as
    legacy_combination,
)

from app.preparation.preparation_combine_service import (
    _require_combine_window,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
    PreparationStageStatus,
)


def fake_plan(
    *action_ids: str,
):
    return SimpleNamespace(
        action_count=
            len(
                action_ids
            ),

        actions=[
            SimpleNamespace(
                action_id=
                    action_id
            )

            for action_id
            in action_ids
        ],
    )


def fake_session(
    *,
    clean_status:
        PreparationStageStatus,

    transform_status:
        PreparationStageStatus,

    validate_status:
        PreparationStageStatus =
            PreparationStageStatus.NOT_STARTED,
):
    return SimpleNamespace(
        snapshot=
            SimpleNamespace(
                stages=[
                    SimpleNamespace(
                        stage=
                            PreparationStage.CLEAN,

                        status=
                            clean_status,
                    ),

                    SimpleNamespace(
                        stage=
                            PreparationStage.TRANSFORM,

                        status=
                            transform_status,
                    ),

                    SimpleNamespace(
                        stage=
                            PreparationStage.VALIDATE,

                        status=
                            validate_status,
                    ),
                ]
            )
    )


def expect_value_error(
    callback,
    expected_text: str,
) -> None:
    try:
        callback()

    except ValueError as error:
        assert (
            expected_text
            in str(
                error
            )
        )

        return

    raise AssertionError(
        (
            "Expected ValueError containing "
            f"{expected_text!r}."
        )
    )


def test_complete_partition() -> None:
    plan = fake_plan(
        "clean:1",
        "clean:2",
        "clean:3",
    )

    _require_complete_cleaning_decision(
        cleaning_plan=
            plan,

        approved_action_ids={
            "clean:1",
        },

        rejected_action_ids={
            "clean:2",
            "clean:3",
        },
    )

    print(
        "[PASS] approved + rejected covers plan"
    )


def test_reject_all() -> None:
    plan = fake_plan(
        "clean:1",
        "clean:2",
    )

    _require_complete_cleaning_decision(
        cleaning_plan=
            plan,

        approved_action_ids=
            set(),

        rejected_action_ids={
            "clean:1",
            "clean:2",
        },
    )

    print(
        "[PASS] reject-all is valid"
    )


def test_missing_decision() -> None:
    plan = fake_plan(
        "clean:1",
        "clean:2",
    )

    expect_value_error(
        lambda:
            _require_complete_cleaning_decision(
                cleaning_plan=
                    plan,

                approved_action_ids={
                    "clean:1",
                },

                rejected_action_ids=
                    set(),
            ),

        "Missing decision",
    )

    print(
        "[PASS] missing decision fails closed"
    )


def test_overlap() -> None:
    plan = fake_plan(
        "clean:1",
    )

    expect_value_error(
        lambda:
            _require_complete_cleaning_decision(
                cleaning_plan=
                    plan,

                approved_action_ids={
                    "clean:1",
                },

                rejected_action_ids={
                    "clean:1",
                },
            ),

        "both approved and rejected",
    )

    print(
        "[PASS] overlap fails closed"
    )


def test_unknown_ids() -> None:
    plan = fake_plan(
        "clean:1",
    )

    expect_value_error(
        lambda:
            _require_complete_cleaning_decision(
                cleaning_plan=
                    plan,

                approved_action_ids={
                    "clean:999",
                },

                rejected_action_ids={
                    "clean:1",
                },
            ),

        "Unknown approved",
    )

    expect_value_error(
        lambda:
            _require_complete_cleaning_decision(
                cleaning_plan=
                    plan,

                approved_action_ids={
                    "clean:1",
                },

                rejected_action_ids={
                    "clean:999",
                },
            ),

        "Unknown rejected",
    )

    print(
        "[PASS] unknown IDs fail closed"
    )


def test_apply_contract() -> None:
    signature = inspect.signature(
        apply_uploaded_cleaning_plan
    )

    assert (
        "approved_action_ids_json"
        in signature.parameters
    )

    assert (
        "rejected_action_ids_json"
        in signature.parameters
    )

    print(
        "[PASS] cleaning-apply exposes full decision contract"
    )


def test_controlled_combine_guard() -> None:
    unresolved = fake_session(
        clean_status=
            PreparationStageStatus.REVIEW_REQUIRED,

        transform_status=
            PreparationStageStatus.PASSED,
    )

    expect_value_error(
        lambda:
            _require_combine_window(
                unresolved
            ),

        "CLEAN",
    )


    resolved = fake_session(
        clean_status=
            PreparationStageStatus.PASSED,

        transform_status=
            PreparationStageStatus.SKIPPED,
    )

    _require_combine_window(
        resolved
    )

    print(
        "[PASS] controlled Combine requires resolved CLEAN"
    )


def test_legacy_combine_guard() -> None:
    unresolved = fake_session(
        clean_status=
            PreparationStageStatus.REVIEW_REQUIRED,

        transform_status=
            PreparationStageStatus.SKIPPED,
    )

    original = (
        legacy_combination
        .get_preparation_session
    )

    try:
        legacy_combination.get_preparation_session = (
            lambda workflow_id:
                unresolved
        )

        try:
            legacy_combination._require_combination_precondition(
                workflow_id=
                    "prep:test"
            )

        except HTTPException as error:
            assert (
                error.status_code
                ==
                409
            )

            assert (
                error.detail[
                    "error"
                ]
                ==
                "clean_stage_not_resolved"
            )

        else:
            raise AssertionError(
                (
                    "Legacy Combination bypassed "
                    "unresolved CLEAN."
                )
            )


        resolved = fake_session(
            clean_status=
                PreparationStageStatus.SKIPPED,

            transform_status=
                PreparationStageStatus.PASSED,
        )

        legacy_combination.get_preparation_session = (
            lambda workflow_id:
                resolved
        )

        returned = (
            legacy_combination
            ._require_combination_precondition(
                workflow_id=
                    "prep:test"
            )
        )

        assert (
            returned
            is resolved
        )

    finally:
        legacy_combination.get_preparation_session = (
            original
        )

    print(
        "[PASS] legacy Combination requires resolved CLEAN"
    )


def main() -> None:
    print()
    print(
        "=== DATALENS CLEAN DECISION / "
        "COMBINE GUARD v0.1 ==="
    )
    print()

    test_complete_partition()
    test_reject_all()
    test_missing_decision()
    test_overlap()
    test_unknown_ids()
    test_apply_contract()
    test_controlled_combine_guard()
    test_legacy_combine_guard()

    print()
    print(
        "PASS - clean decision / "
        "Combine guard v0.1"
    )


if __name__ == "__main__":
    main()
