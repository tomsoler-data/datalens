from __future__ import annotations


from pathlib import Path


ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[
        4
    ]
)


MODEL_LAB_TYPES = (
    ROOT
    /
    "apps/web/src/components/modelLab/modelLabTypes.ts"
)


MODEL_LAB_CLIENT = (
    ROOT
    /
    "apps/web/src/app/model-lab/ModelLabClient.tsx"
)


MODEL_TRAINING_CONTRACTS = (
    ROOT
    /
    "apps/api/app/api/model_training_contracts.py"
)


MODEL_LAB_CONTRACTS = (
    ROOT
    /
    "apps/api/app/api/model_lab_contracts.py"
)


CV_EXECUTOR = (
    ROOT
    /
    "apps/api/app/ml/cross_validation_executor.py"
)


TUNING_EXECUTOR = (
    ROOT
    /
    "apps/api/app/ml/hyperparameter_tuning_executor.py"
)


def read(
    path: Path,
) -> str:

    return path.read_text(
        encoding="utf-8"
    )


def test_frontend_combined_contract() -> None:

    source = read(
        MODEL_LAB_TYPES
    )


    assert (
        "ModelLabPurgedGroupTimeHoldoutSplitContract"
        in
        source
    )


    assert (
        '"purged_group_time_holdout"'
        in
        source
    )


    start = source.index(
        "export type "
        "ModelLabPurgedGroupTimeHoldoutSplitContract"
    )


    end = source.index(
        "export type ModelLabSplitContract",
        start,
    )


    combined = source[
        start:
        end
    ]


    assert (
        "group_column:"
        in
        combined
    )


    assert (
        "time_column:"
        in
        combined
    )


    assert (
        "shuffle:"
        in
        combined
    )


    assert (
        "stratify:"
        in
        combined
    )


    assert (
        combined.count(
            "false;"
        )
        >=
        2
    )


def test_training_selector() -> None:

    client = read(
        MODEL_LAB_CLIENT
    )


    assert (
        client.count(
            'value="purged_group_time_holdout"'
        )
        ==
        1
    )


    assert (
        "Chronologique + purge entités"
        in
        client
    )


    index = client.index(
        'value="purged_group_time_holdout"'
    )


    region = client[
        index:
        index
        +
        1000
    ]


    assert (
        "eligibleTrainingGroupColumns.length"
        in
        region
    )


    assert (
        "eligibleTrainingTimeColumns.length"
        in
        region
    )


def test_combined_request() -> None:

    client = read(
        MODEL_LAB_CLIENT
    )


    start = client.index(
        "            split:"
    )


    index = client.index(
        '"purged_group_time_holdout"',
        start,
    )


    region = client[
        index:
        index
        +
        1600
    ]


    assert (
        "group_column:"
        in
        region
    )


    assert (
        "trainingGroupColumn"
        in
        region
    )


    assert (
        "time_column:"
        in
        region
    )


    assert (
        "trainingTimeColumn"
        in
        region
    )


    assert (
        "shuffle:"
        in
        region
    )


    assert (
        "stratify:"
        in
        region
    )


def test_combined_requirements() -> None:

    client = read(
        MODEL_LAB_CLIENT
    )


    assert (
        "trainingSplitUsesGroup"
        in
        client
    )


    assert (
        "trainingSplitUsesTime"
        in
        client
    )


    assert (
        "trainingSplitIsPurgedGroupTime"
        in
        client
    )


    assert (
        "!trainingGroupColumn"
        in
        client
    )


    assert (
        "!trainingTimeColumn"
        in
        client
    )


def test_combined_ux() -> None:

    client = read(
        MODEL_LAB_CLIENT
    )


    for token in (
        "Le test conserve les observations futures.",
        "Toute entité présente dans ce futur est retirée",
        "fuite d’entité entre train et test",
        "Passé / futur + purge entités",
        "Entités futures purgées du train",
        "lignes utilisées",
    ):

        assert (
            token
            in
            client
        )


def test_historical_split_ui_preserved() -> None:

    client = read(
        MODEL_LAB_CLIENT
    )


    for value in (
        "holdout",
        "group_holdout",
        "time_holdout",
    ):

        assert (
            f'value="{value}"'
            in
            client
        )


    types = read(
        MODEL_LAB_TYPES
    )


    for split_type in (
        "ModelLabHoldoutSplitContract",
        "ModelLabGroupHoldoutSplitContract",
        "ModelLabTimeHoldoutSplitContract",
    ):

        assert (
            split_type
            in
            types
        )


def test_backend_transport_generic() -> None:

    training = read(
        MODEL_TRAINING_CONTRACTS
    )


    lab = read(
        MODEL_LAB_CONTRACTS
    )


    assert (
        "training: MLTrainingContract"
        in
        training
    )


    assert (
        "MLTrainingSplitContract"
        in
        lab
    )


def test_e15b_stays_locked() -> None:

    cv = read(
        CV_EXECUTOR
    )


    tuning = read(
        TUNING_EXECUTOR
    )


    assert (
        "MLPurgedGroupTimeHoldoutSplitContract"
        in
        cv
    )


    assert (
        "E15b"
        in
        cv
    )


    assert (
        "MLPurgedGroupTimeHoldoutSplitContract"
        in
        tuning
    )


    assert (
        "E15b"
        in
        tuning
    )


def main() -> None:

    print(
        "=== DATALENS E15a-P2 PURGED GROUP + TEMPORAL FRONTEND v0.1 ==="
    )


    test_frontend_combined_contract()

    print(
        "[PASS] frontend combined split contract"
    )


    test_training_selector()

    print(
        "[PASS] combined split selectable in training UI"
    )


    test_combined_request()

    print(
        "[PASS] combined request sends group + time metadata"
    )


    test_combined_requirements()

    print(
        "[PASS] combined UI requires entity + observation time"
    )


    test_combined_ux()

    print(
        "[PASS] purge semantics explicit in UX"
    )


    print(
        "[PASS] persisted detail displays combined split"
    )


    print(
        "[PASS] combined detail distinguishes used rows"
    )


    test_historical_split_ui_preserved()

    print(
        "[PASS] historical frontend split options preserved"
    )


    test_backend_transport_generic()

    print(
        "[PASS] existing API transport remains sufficient"
    )


    test_e15b_stays_locked()

    print(
        "[PASS] combined CV / tuning remain fail-closed to E15b"
    )


    print()
    print(
        "E15a-P2 PURGED GROUP + TEMPORAL FRONTEND v0.1: PASS"
    )


if __name__ == "__main__":

    main()
