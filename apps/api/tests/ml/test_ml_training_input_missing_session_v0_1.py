from __future__ import annotations


from types import (
    SimpleNamespace,
)


from app.ml.training_input import (
    MLTrainingInputError,
    load_authorized_ml_dataframe,
)


from app.preparation.preparation_session import (
    PreparationSessionNotFoundError,
)


# ============================================================
# FIXTURE
# ============================================================


WORKFLOW_ID = (
    "workflow:missing-preparation-session"
)


DATASET_ID = (
    "dataset:missing-preparation-session"
)


contract = (
    SimpleNamespace(
        workflow_id=
            WORKFLOW_ID,

        dataset_id=
            DATASET_ID,
    )
)


loader_called = {
    "value":
        False,
}


def missing_handoff_loader(
    *,
    workflow_id: str,
):

    loader_called[
        "value"
    ] = True


    assert (
        workflow_id
        ==
        WORKFLOW_ID
    )


    raise PreparationSessionNotFoundError(
        (
            "Preparation session not found: "
            f"{workflow_id}"
        )
    )


# ============================================================
# EXPECTED NORMALIZATION
# ============================================================


try:

    load_authorized_ml_dataframe(
        contract=
            contract,

        handoff_loader=
            missing_handoff_loader,

        execution_label=
            "Controlled Model Lab",
    )

except MLTrainingInputError as error:

    assert (
        loader_called[
            "value"
        ]
        is True
    )


    assert isinstance(
        error.__cause__,
        PreparationSessionNotFoundError,
    )


    assert (
        "Preparation session not found"
        not in
        str(
            error
        )
    )


    assert (
        "Preparation did not provide a valid READY"
        in
        str(
            error
        )
    )


else:

    raise AssertionError(
        (
            "Missing Preparation session must be "
            "normalized to MLTrainingInputError."
        )
    )


print(
    "[PASS] missing Preparation session normalizes to MLTrainingInputError"
)


print()
print("=" * 80)
print("ML TRAINING INPUT MISSING SESSION REGRESSION VERDICT")
print("=" * 80)
print()

print("PreparationSessionNotFoundError caught        PASS")
print("MLTrainingInputError boundary                 PASS")
print("Original exception retained as cause          PASS")
print("Preparation internal detail not exposed       PASS")

print()
print(
    "ML training input missing-session normalization: PASS"
)
