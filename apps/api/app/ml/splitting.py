from __future__ import annotations


import math


from dataclasses import (
    dataclass,
)


from typing import (
    Protocol,
)


import numpy as np
import pandas as pd


from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)


from app.ml.contracts import (
    MLGroupHoldoutSplitContract,
    MLPurgedGroupTimeHoldoutSplitContract,
    MLSplitContract,
    MLTimeHoldoutSplitContract,
    MLTrainingContract,
    MLTrainingSplitContract,
)


from app.profiling.types import (
    infer_analytical_type,
)


# ============================================================
# VERSION
# ============================================================


ML_SPLITTING_RULE_VERSION = (
    "ml_splitting_v0.1"
)


ML_HOLDOUT_PARTITION_RULE_VERSION = (
    "ml_holdout_partition_v0.1"
)


ML_FEATURE_HOLDOUT_PARTITION_RULE_VERSION = (
    "ml_feature_holdout_partition_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLSplitError(
    RuntimeError
):
    pass


class MLSplitInputError(
    MLSplitError
):
    pass


class MLSplitInvariantError(
    MLSplitError
):
    pass


class MLHoldoutContract(
    Protocol
):
    """
    Structural contract required by shared holdout-position
    resolution.

    Both supervised and anomaly contracts provide this split
    surface without coupling the resolver to model semantics.
    """

    split: MLTrainingSplitContract

# ============================================================
# SHARED MODEL LAB SPLIT AUTHORITY
# ============================================================


def validated_group_values(
    *,
    dataframe: pd.DataFrame | None,
    x: pd.DataFrame,
    y: pd.Series | None,
    contract: MLHoldoutContract,
) -> pd.Series:

    split = (
        contract.split
    )


    if not isinstance(
        split,
        (
            MLGroupHoldoutSplitContract,
            MLPurgedGroupTimeHoldoutSplitContract,
        ),
    ):

        raise (
            MLSplitInputError(
                (
                    "Group validation requires a "
                    "group-aware holdout split contract."
                )
            )
        )


    if dataframe is None:

        raise (
            MLSplitInputError(
                (
                    "group_holdout requires the "
                    "server-owned source dataframe."
                )
            )
        )


    target_alignment_invalid = (
        y is not None
        and
        (
            not isinstance(
                y,
                pd.Series,
            )
            or
            len(
                x
            )
            !=
            len(
                y
            )
            or
            not x.index.equals(
                y.index
            )
        )
    )


    if (
        len(
            dataframe
        )
        !=
        len(
            x
        )
        or
        not dataframe.index.equals(
            x.index
        )
        or
        target_alignment_invalid
    ):

        raise (
            MLSplitInputError(
                (
                    "Entity-aware split input "
                    "alignment is invalid."
                )
            )
        )


    group_column = (
        split.group_column
    )


    if (
        group_column
        not in
        dataframe.columns
    ):

        raise (
            MLSplitInputError(
                (
                    "Group holdout column is "
                    "missing from the validated "
                    "dataset. "
                    f"group_column={group_column}"
                )
            )
        )


    group_values = (
        dataframe.loc[
            :,
            group_column,
        ]
    )


    if not isinstance(
        group_values,
        pd.Series,
    ):

        raise (
            MLSplitInputError(
                (
                    "Group holdout column must "
                    "resolve to exactly one Series."
                )
            )
        )


    group_values = (
        group_values.copy(
            deep=True
        )
    )


    if bool(
        group_values
        .isna()
        .any()
    ):

        raise (
            MLSplitInputError(
                (
                    "Group holdout column contains "
                    "missing values. "
                    f"group_column={group_column}"
                )
            )
        )


    semantics = (
        infer_analytical_type(
            group_column,
            group_values,
        )
    )


    analytical_type = str(
        semantics.get(
            "type"
        )
        or
        ""
    ).strip()


    analytical_subtype = str(
        semantics.get(
            "subtype"
        )
        or
        ""
    ).strip()


    if (
        analytical_type
        !=
        "identifier"
        or
        analytical_subtype
        !=
        "reference"
    ):

        raise (
            MLSplitInputError(
                (
                    "Group holdout requires a "
                    "repeated reference identifier. "
                    f"group_column={group_column}, "
                    f"analytical_type={analytical_type}, "
                    "analytical_subtype="
                    f"{analytical_subtype}"
                )
            )
        )


    group_count = int(
        group_values
        .nunique(
            dropna=True
        )
    )


    if group_count < 2:

        raise (
            MLSplitInputError(
                (
                    "Group holdout requires at "
                    "least two distinct entity groups."
                )
            )
        )


    if (
        group_count
        >=
        len(
            group_values
        )
    ):

        raise (
            MLSplitInputError(
                (
                    "A unique row identifier cannot "
                    "be used as the entity holdout "
                    "group."
                )
            )
        )


    return group_values



def validated_time_values(
    *,
    dataframe: pd.DataFrame | None,
    x: pd.DataFrame,
    y: pd.Series | None,
    contract: MLHoldoutContract,
) -> pd.Series:

    split = (
        contract.split
    )


    if not isinstance(
        split,
        (
            MLTimeHoldoutSplitContract,
            MLPurgedGroupTimeHoldoutSplitContract,
        ),
    ):

        raise (
            MLSplitInputError(
                (
                    "Time validation requires a "
                    "time-aware holdout split contract."
                )
            )
        )


    if dataframe is None:

        raise (
            MLSplitInputError(
                (
                    "time_holdout requires the "
                    "server-owned source dataframe."
                )
            )
        )


    target_alignment_invalid = (
        y is not None
        and
        (
            not isinstance(
                y,
                pd.Series,
            )
            or
            len(
                x
            )
            !=
            len(
                y
            )
            or
            not x.index.equals(
                y.index
            )
        )
    )


    if (
        len(
            dataframe
        )
        !=
        len(
            x
        )
        or
        not dataframe.index.equals(
            x.index
        )
        or
        target_alignment_invalid
    ):

        raise (
            MLSplitInputError(
                (
                    "Time-aware split input "
                    "alignment is invalid."
                )
            )
        )


    time_column = (
        split.time_column
    )


    if (
        time_column
        not in
        dataframe.columns
    ):

        raise (
            MLSplitInputError(
                (
                    "Time holdout column is "
                    "missing from the validated "
                    "dataset. "
                    f"time_column={time_column}"
                )
            )
        )


    time_values = (
        dataframe.loc[
            :,
            time_column,
        ]
    )


    if not isinstance(
        time_values,
        pd.Series,
    ):

        raise (
            MLSplitInputError(
                (
                    "Time holdout column must "
                    "resolve to exactly one Series."
                )
            )
        )


    time_values = (
        time_values.copy(
            deep=True
        )
    )


    if bool(
        time_values
        .isna()
        .any()
    ):

        raise (
            MLSplitInputError(
                (
                    "Time holdout column contains "
                    "missing values. "
                    f"time_column={time_column}"
                )
            )
        )


    if not (
        pd.api.types
        .is_datetime64_any_dtype(
            time_values.dtype
        )
    ):

        raise (
            MLSplitInputError(
                (
                    "Time holdout requires a "
                    "validated pandas datetime "
                    "column. Model Lab does not "
                    "implicitly parse string dates. "
                    f"time_column={time_column}, "
                    f"dtype={time_values.dtype}"
                )
            )
        )


    semantics = (
        infer_analytical_type(
            time_column,
            time_values,
        )
    )


    analytical_type = str(
        semantics.get(
            "type"
        )
        or
        ""
    ).strip()


    analytical_subtype = str(
        semantics.get(
            "subtype"
        )
        or
        ""
    ).strip()


    if (
        analytical_type
        !=
        "temporal"
        or
        analytical_subtype
        !=
        "datetime"
    ):

        raise (
            MLSplitInputError(
                (
                    "Time holdout requires "
                    "observation-time datetime "
                    "semantics. "
                    f"time_column={time_column}, "
                    f"analytical_type={analytical_type}, "
                    "analytical_subtype="
                    f"{analytical_subtype}"
                )
            )
        )


    distinct_timestamps = int(
        time_values
        .nunique(
            dropna=True
        )
    )


    if distinct_timestamps < 2:

        raise (
            MLSplitInputError(
                (
                    "Time holdout requires at "
                    "least two distinct timestamps."
                )
            )
        )


    return time_values



def chronological_holdout_positions(
    *,
    time_values: pd.Series,
    test_size: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Resolve the deterministic chronological OUTER boundary.

    Equal timestamps are never split across TRAIN and TEST.

    time_holdout and purged_group_time_holdout deliberately
    reuse this exact authority.
    """

    row_count = int(
        len(
            time_values
        )
    )


    desired_test_rows = max(
        2,
        int(
            math.ceil(
                row_count
                *
                test_size
            )
        ),
    )


    initial_cut_position = (
        row_count
        -
        desired_test_rows
    )


    if initial_cut_position < 2:

        raise (
            MLSplitInputError(
                (
                    "Time holdout cannot preserve "
                    "at least two training rows "
                    "with the configured test_size."
                )
            )
        )


    ordered_positions = (
        np.argsort(
            time_values.to_numpy(),
            kind="stable",
        )
    )


    ordered_times = (
        time_values.iloc[
            ordered_positions
        ]
        .reset_index(
            drop=True
        )
    )


    cut_position = (
        initial_cut_position
    )


    boundary_timestamp = (
        ordered_times.iloc[
            cut_position
        ]
    )


    while (
        cut_position
        >
        0
        and
        ordered_times.iloc[
            cut_position
            -
            1
        ]
        ==
        boundary_timestamp
    ):

        cut_position -= 1


    if cut_position < 2:

        raise (
            MLSplitInputError(
                (
                    "Time holdout timestamp "
                    "boundary would leave fewer "
                    "than two training rows. "
                    "Equal timestamps are never "
                    "split across train and test."
                )
            )
        )


    train_positions = (
        ordered_positions[
            :cut_position
        ]
    )


    test_positions = (
        ordered_positions[
            cut_position:
        ]
    )


    if (
        len(
            test_positions
        )
        <
        2
    ):

        raise (
            MLSplitInputError(
                (
                    "Time holdout produced "
                    "fewer than two test rows."
                )
            )
        )


    return (
        train_positions,
        test_positions,
    )


def split_ml_dataset(
    *,
    x: pd.DataFrame,
    y: pd.Series,
    contract: MLTrainingContract,
    dataframe: pd.DataFrame | None = None,
    return_group_partitions: bool = False,
    return_time_partitions: bool = False,
) -> (
    tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
    ]
    |
    tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        pd.Series | None,
        pd.Series | None,
    ]
    |
    tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        pd.Series | None,
        pd.Series | None,
        pd.Series | None,
        pd.Series | None,
    ]
):

    split = (
        contract.split
    )


    train_groups: pd.Series | None = None

    test_groups: pd.Series | None = None


    train_times: pd.Series | None = None

    test_times: pd.Series | None = None


    if (
        return_group_partitions
        and
        return_time_partitions
        and
        not isinstance(
            split,
            MLPurgedGroupTimeHoldoutSplitContract,
        )
    ):

        raise (
            MLSplitInputError(
                (
                    "Group and time split metadata "
                    "cannot be requested together "
                    "outside purged_group_time_holdout."
                )
            )
        )


    # ========================================================
    # PURGED GROUP + TEMPORAL HOLDOUT
    #
    # 1. Create future TEST from the chronological boundary.
    # 2. Identify every entity present in future TEST.
    # 3. Purge historical observations belonging to those
    #    entities from TRAIN.
    #
    # TEST remains unchanged.
    # ========================================================


    if isinstance(
        split,
        MLPurgedGroupTimeHoldoutSplitContract,
    ):

        groups = (
            validated_group_values(
                dataframe=
                    dataframe,

                x=
                    x,

                y=
                    y,

                contract=
                    contract,
            )
        )


        time_values = (
            validated_time_values(
                dataframe=
                    dataframe,

                x=
                    x,

                y=
                    y,

                contract=
                    contract,
            )
        )


        (
            candidate_train_positions,
            test_positions,
        ) = (
            chronological_holdout_positions(
                time_values=
                    time_values,

                test_size=
                    split.test_size,
            )
        )


        candidate_train_groups = (
            groups.iloc[
                candidate_train_positions
            ]
            .copy(
                deep=True
            )
        )


        test_groups = (
            groups.iloc[
                test_positions
            ]
            .copy(
                deep=True
            )
        )


        test_group_values = set(
            test_groups.tolist()
        )


        if not test_group_values:

            raise (
                MLSplitInputError(
                    (
                        "Purged group + temporal "
                        "holdout produced no future "
                        "test entity groups."
                    )
                )
            )


        keep_train_mask = (
            ~candidate_train_groups
            .isin(
                test_group_values
            )
        ).to_numpy(
            dtype=bool
        )


        train_positions = (
            candidate_train_positions[
                keep_train_mask
            ]
        )


        if (
            len(
                train_positions
            )
            <
            2
        ):

            raise (
                MLSplitInputError(
                    (
                        "Purging future-test entity "
                        "groups from the historical "
                        "training candidate left fewer "
                        "than two training rows."
                    )
                )
            )


        x_train = (
            x.iloc[
                train_positions
            ]
            .copy(
                deep=True
            )
        )


        x_test = (
            x.iloc[
                test_positions
            ]
            .copy(
                deep=True
            )
        )


        y_train = (
            y.iloc[
                train_positions
            ]
            .copy(
                deep=True
            )
        )


        y_test = (
            y.iloc[
                test_positions
            ]
            .copy(
                deep=True
            )
        )


        train_groups = (
            groups.iloc[
                train_positions
            ]
            .copy(
                deep=True
            )
        )


        test_groups = (
            groups.iloc[
                test_positions
            ]
            .copy(
                deep=True
            )
        )


        train_times = (
            time_values.iloc[
                train_positions
            ]
            .copy(
                deep=True
            )
        )


        test_times = (
            time_values.iloc[
                test_positions
            ]
            .copy(
                deep=True
            )
        )


        train_group_values = set(
            train_groups.tolist()
        )


        test_group_values = set(
            test_groups.tolist()
        )


        if not train_group_values:

            raise (
                MLSplitInputError(
                    (
                        "Purged group + temporal "
                        "holdout produced no training "
                        "entity groups."
                    )
                )
            )


        if (
            train_group_values
            &
            test_group_values
        ):

            raise (
                MLSplitInvariantError(
                    (
                        "Purged group + temporal "
                        "holdout produced overlapping "
                        "train/test entity groups."
                    )
                )
            )


        if not (
            train_times.max()
            <
            test_times.min()
        ):

            raise (
                MLSplitInvariantError(
                    (
                        "Purged group + temporal "
                        "holdout violated the strict "
                        "chronological boundary."
                    )
                )
            )


        if (
            set(
                train_times.tolist()
            )
            &
            set(
                test_times.tolist()
            )
        ):

            raise (
                MLSplitInvariantError(
                    (
                        "Purged group + temporal "
                        "holdout produced overlapping "
                        "train/test timestamps."
                    )
                )
            )


    # ========================================================
    # ENTITY-AWARE HOLDOUT
    # ========================================================


    elif isinstance(
        split,
        MLGroupHoldoutSplitContract,
    ):

        groups = (
            validated_group_values(
                dataframe=
                    dataframe,

                x=
                    x,

                y=
                    y,

                contract=
                    contract,
            )
        )


        splitter = (
            GroupShuffleSplit(
                n_splits=
                    1,

                test_size=
                    split.test_size,

                random_state=
                    split.random_seed,
            )
        )


        try:

            (
                train_indices,
                test_indices,
            ) = next(
                splitter.split(
                    x,
                    y,
                    groups=
                        groups,
                )
            )


        except ValueError as error:

            raise (
                MLSplitInputError(
                    (
                        "Deterministic entity-aware "
                        "train/test split could not "
                        "be created from the ML "
                        "Training Contract."
                    )
                )
            ) from error


        x_train = (
            x.iloc[
                train_indices
            ]
            .copy(
                deep=True
            )
        )


        x_test = (
            x.iloc[
                test_indices
            ]
            .copy(
                deep=True
            )
        )


        y_train = (
            y.iloc[
                train_indices
            ]
            .copy(
                deep=True
            )
        )


        y_test = (
            y.iloc[
                test_indices
            ]
            .copy(
                deep=True
            )
        )


        train_groups = (
            groups.iloc[
                train_indices
            ]
        )


        test_groups = (
            groups.iloc[
                test_indices
            ]
        )


        train_group_values = set(
            train_groups.tolist()
        )


        test_group_values = set(
            test_groups.tolist()
        )


        if not train_group_values:

            raise (
                MLSplitInputError(
                    (
                        "Entity-aware split produced "
                        "no training entity groups."
                    )
                )
            )


        if not test_group_values:

            raise (
                MLSplitInputError(
                    (
                        "Entity-aware split produced "
                        "no test entity groups."
                    )
                )
            )


        overlap = (
            train_group_values
            &
            test_group_values
        )


        if overlap:

            raise (
                MLSplitInvariantError(
                    (
                        "Entity-aware split produced "
                        "overlapping train/test groups."
                    )
                )
            )


    # ========================================================
    # TEMPORAL HOLDOUT
    # ========================================================


    elif isinstance(
        split,
        MLTimeHoldoutSplitContract,
    ):

        time_values = (
            validated_time_values(
                dataframe=
                    dataframe,

                x=
                    x,

                y=
                    y,

                contract=
                    contract,
            )
        )


        (
            train_positions,
            test_positions,
        ) = (
            chronological_holdout_positions(
                time_values=
                    time_values,

                test_size=
                    split.test_size,
            )
        )


        x_train = (
            x.iloc[
                train_positions
            ]
            .copy(
                deep=True
            )
        )


        x_test = (
            x.iloc[
                test_positions
            ]
            .copy(
                deep=True
            )
        )


        y_train = (
            y.iloc[
                train_positions
            ]
            .copy(
                deep=True
            )
        )


        y_test = (
            y.iloc[
                test_positions
            ]
            .copy(
                deep=True
            )
        )


        train_times = (
            time_values.iloc[
                train_positions
            ]
            .copy(
                deep=True
            )
        )


        test_times = (
            time_values.iloc[
                test_positions
            ]
            .copy(
                deep=True
            )
        )


        train_max_time = (
            train_times.max()
        )


        test_min_time = (
            test_times.min()
        )


        if not (
            train_max_time
            <
            test_min_time
        ):

            raise (
                MLSplitInvariantError(
                    (
                        "Time holdout violated the "
                        "strict chronological boundary. "
                        "Every training timestamp must "
                        "be earlier than every test "
                        "timestamp."
                    )
                )
            )


        timestamp_overlap = (
            set(
                train_times.tolist()
            )
            &
            set(
                test_times.tolist()
            )
        )


        if timestamp_overlap:

            raise (
                MLSplitInvariantError(
                    (
                        "Time holdout produced "
                        "overlapping timestamps across "
                        "train and test."
                    )
                )
            )


    # ========================================================
    # HISTORICAL ROW HOLDOUT
    # ========================================================


    elif isinstance(
        split,
        MLSplitContract,
    ):

        if (
            split.stratify
            and
            not split.shuffle
        ):

            raise (
                MLSplitInputError(
                    (
                        "stratify=True requires "
                        "shuffle=True for the "
                        "Classical ML holdout split."
                    )
                )
            )


        stratify_values = (
            y

            if (
                contract.problem_type
                ==
                "classification"
                and
                split.stratify
            )

            else None
        )


        random_state = (
            split.random_seed

            if split.shuffle

            else None
        )


        try:

            (
                x_train,
                x_test,
                y_train,
                y_test,
            ) = (
                train_test_split(
                    x,
                    y,

                    test_size=
                        split.test_size,

                    random_state=
                        random_state,

                    shuffle=
                        split.shuffle,

                    stratify=
                        stratify_values,
                )
            )


        except ValueError as error:

            raise (
                MLSplitInputError(
                    (
                        "Deterministic train/test "
                        "split could not be created "
                        "from the ML Training Contract."
                    )
                )
            ) from error


    else:

        raise (
            MLSplitInputError(
                "Unsupported ML split contract."
            )
        )


    # ========================================================
    # COMMON HOLDOUT INVARIANTS
    # ========================================================


    if (
        len(
            x_train
        )
        <
        2
        or
        len(
            x_test
        )
        <
        2
    ):

        raise (
            MLSplitInputError(
                (
                    "Classical ML v0.1 requires "
                    "at least two training rows "
                    "and two test rows after "
                    "the holdout split."
                )
            )
        )


    if (
        contract.problem_type
        ==
        "classification"
        and
        int(
            y_train.nunique(
                dropna=False
            )
        )
        <
        2
    ):

        raise (
            MLSplitInputError(
                (
                    "Classification training split "
                    "contains fewer than two "
                    "classes."
                )
            )
        )


    if (
        return_group_partitions
        and
        return_time_partitions
    ):

        if (
            train_groups
            is None
            or
            test_groups
            is None
            or
            train_times
            is None
            or
            test_times
            is None
        ):

            raise (
                MLSplitInputError(
                    (
                        "Combined split metadata was "
                        "requested but the validated "
                        "group/time partitions are "
                        "incomplete."
                    )
                )
            )


        return (
            x_train,
            x_test,
            y_train,
            y_test,
            train_groups.copy(
                deep=True
            ),
            test_groups.copy(
                deep=True
            ),
            train_times.copy(
                deep=True
            ),
            test_times.copy(
                deep=True
            ),
        )


    if return_time_partitions:

        if (
            train_times
            is None
            or
            test_times
            is None
        ):

            raise (
                MLSplitInputError(
                    (
                        "Temporal split metadata was "
                        "requested for a non-temporal "
                        "holdout."
                    )
                )
            )


        return (
            x_train,
            x_test,
            y_train,
            y_test,
            train_times.copy(
                deep=True
            ),
            test_times.copy(
                deep=True
            ),
        )


    if return_group_partitions:

        return (
            x_train,
            x_test,
            y_train,
            y_test,
            (
                train_groups.copy(
                    deep=True
                )

                if train_groups
                is not None

                else None
            ),
            (
                test_groups.copy(
                    deep=True
                )

                if test_groups
                is not None

                else None
            ),
        )


    return (
        x_train,
        x_test,
        y_train,
        y_test,
    )


# ============================================================
# EXPLICIT HOLDOUT POSITION AUTHORITY
# ============================================================


@dataclass(
    frozen=True,
)
class MLHoldoutPartition:
    """
    Exact server-owned row positions used by one Model Lab
    outer holdout.

    Positions are relative to the validated source row order.

    They are deliberately independent of pandas index labels,
    which may be non-consecutive or duplicated.
    """

    source_row_count: int

    train_positions: tuple[
        int,
        ...
    ]

    test_positions: tuple[
        int,
        ...
    ]

    purged_positions: tuple[
        int,
        ...
    ] = ()


    @property
    def train_rows(
        self,
    ) -> int:
        return len(
            self.train_positions
        )


    @property
    def test_rows(
        self,
    ) -> int:
        return len(
            self.test_positions
        )


    @property
    def purged_rows(
        self,
    ) -> int:
        return len(
            self.purged_positions
        )


def resolve_ml_feature_holdout_partition(
    *,
    x: pd.DataFrame,
    contract: MLHoldoutContract,
    dataframe: pd.DataFrame | None = None,
    stratify_values: pd.Series | None = None,
) -> MLHoldoutPartition:
    """
    Resolve exact outer-holdout row positions from features
    and server-owned split metadata.

    No target is required.

    Supervised callers may explicitly provide stratify_values
    when classification stratification is requested.
    """

    if not isinstance(
        x,
        pd.DataFrame,
    ):
        raise (
            MLSplitInputError(
                "ML holdout features must be a pandas DataFrame."
            )
        )


    source_row_count = int(
        len(
            x
        )
    )


    positional_x = (
        x.copy(
            deep=True
        )
        .reset_index(
            drop=True
        )
    )


    positional_dataframe: (
        pd.DataFrame
        |
        None
    ) = None


    if dataframe is not None:

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise (
                MLSplitInputError(
                    (
                        "ML holdout source dataframe must "
                        "be a pandas DataFrame."
                    )
                )
            )


        if (
            len(
                dataframe
            )
            !=
            source_row_count
            or
            not dataframe.index.equals(
                x.index
            )
        ):
            raise (
                MLSplitInputError(
                    (
                        "ML holdout source dataframe must "
                        "be row-aligned with features."
                    )
                )
            )


        positional_dataframe = (
            dataframe.copy(
                deep=True
            )
            .reset_index(
                drop=True
            )
        )


    positional_stratify: (
        pd.Series
        |
        None
    ) = None


    if stratify_values is not None:

        if (
            not isinstance(
                stratify_values,
                pd.Series,
            )
            or
            len(
                stratify_values
            )
            !=
            source_row_count
            or
            not stratify_values.index.equals(
                x.index
            )
        ):
            raise (
                MLSplitInputError(
                    (
                        "ML holdout stratification values "
                        "must be row-aligned with features."
                    )
                )
            )


        positional_stratify = (
            stratify_values.copy(
                deep=True
            )
            .reset_index(
                drop=True
            )
        )


    split = (
        contract.split
    )


    if (
        positional_stratify
        is not None
        and
        not isinstance(
            split,
            MLSplitContract,
        )
    ):
        raise (
            MLSplitInputError(
                (
                    "Target stratification values are "
                    "only valid for regular holdout."
                )
            )
        )


    train_positions = None
    test_positions = None


    if isinstance(
        split,
        MLPurgedGroupTimeHoldoutSplitContract,
    ):

        groups = (
            validated_group_values(
                dataframe=
                    positional_dataframe,
                x=
                    positional_x,
                y=
                    None,
                contract=
                    contract,
            )
        )


        time_values = (
            validated_time_values(
                dataframe=
                    positional_dataframe,
                x=
                    positional_x,
                y=
                    None,
                contract=
                    contract,
            )
        )


        (
            candidate_train_positions,
            test_positions,
        ) = (
            chronological_holdout_positions(
                time_values=
                    time_values,
                test_size=
                    split.test_size,
            )
        )


        candidate_train_groups = (
            groups.iloc[
                candidate_train_positions
            ]
            .copy(
                deep=True
            )
        )


        test_groups = (
            groups.iloc[
                test_positions
            ]
            .copy(
                deep=True
            )
        )


        test_group_values = set(
            test_groups.tolist()
        )


        if not test_group_values:
            raise (
                MLSplitInputError(
                    (
                        "Purged group + temporal "
                        "holdout produced no future "
                        "test entity groups."
                    )
                )
            )


        keep_train_mask = (
            ~candidate_train_groups
            .isin(
                test_group_values
            )
        ).to_numpy(
            dtype=bool
        )


        train_positions = (
            candidate_train_positions[
                keep_train_mask
            ]
        )


        if (
            len(
                train_positions
            )
            <
            2
        ):
            raise (
                MLSplitInputError(
                    (
                        "Purging future-test entity "
                        "groups from the historical "
                        "training candidate left fewer "
                        "than two training rows."
                    )
                )
            )


        train_groups = (
            groups.iloc[
                train_positions
            ]
        )


        train_times = (
            time_values.iloc[
                train_positions
            ]
        )


        test_times = (
            time_values.iloc[
                test_positions
            ]
        )


        if (
            set(
                train_groups.tolist()
            )
            &
            set(
                test_groups.tolist()
            )
        ):
            raise (
                MLSplitInvariantError(
                    (
                        "Purged group + temporal "
                        "holdout produced overlapping "
                        "train/test entity groups."
                    )
                )
            )


        if not (
            train_times.max()
            <
            test_times.min()
        ):
            raise (
                MLSplitInvariantError(
                    (
                        "Purged group + temporal "
                        "holdout violated the strict "
                        "chronological boundary."
                    )
                )
            )


        if (
            set(
                train_times.tolist()
            )
            &
            set(
                test_times.tolist()
            )
        ):
            raise (
                MLSplitInvariantError(
                    (
                        "Purged group + temporal "
                        "holdout produced overlapping "
                        "train/test timestamps."
                    )
                )
            )


    elif isinstance(
        split,
        MLGroupHoldoutSplitContract,
    ):

        groups = (
            validated_group_values(
                dataframe=
                    positional_dataframe,
                x=
                    positional_x,
                y=
                    None,
                contract=
                    contract,
            )
        )


        splitter = (
            GroupShuffleSplit(
                n_splits=
                    1,
                test_size=
                    split.test_size,
                random_state=
                    split.random_seed,
            )
        )


        try:
            (
                train_positions,
                test_positions,
            ) = next(
                splitter.split(
                    positional_x,
                    groups=
                        groups,
                )
            )

        except ValueError as error:
            raise (
                MLSplitInputError(
                    (
                        "Deterministic entity-aware "
                        "train/test split could not "
                        "be created from the ML "
                        "Training Contract."
                    )
                )
            ) from error


        train_group_values = set(
            groups.iloc[
                train_positions
            ]
            .tolist()
        )


        test_group_values = set(
            groups.iloc[
                test_positions
            ]
            .tolist()
        )


        if not train_group_values:
            raise (
                MLSplitInputError(
                    (
                        "Entity-aware split produced "
                        "no training entity groups."
                    )
                )
            )


        if not test_group_values:
            raise (
                MLSplitInputError(
                    (
                        "Entity-aware split produced "
                        "no test entity groups."
                    )
                )
            )


        if (
            train_group_values
            &
            test_group_values
        ):
            raise (
                MLSplitInvariantError(
                    (
                        "Entity-aware split produced "
                        "overlapping train/test groups."
                    )
                )
            )


    elif isinstance(
        split,
        MLTimeHoldoutSplitContract,
    ):

        time_values = (
            validated_time_values(
                dataframe=
                    positional_dataframe,
                x=
                    positional_x,
                y=
                    None,
                contract=
                    contract,
            )
        )


        (
            train_positions,
            test_positions,
        ) = (
            chronological_holdout_positions(
                time_values=
                    time_values,
                test_size=
                    split.test_size,
            )
        )


        train_times = (
            time_values.iloc[
                train_positions
            ]
        )


        test_times = (
            time_values.iloc[
                test_positions
            ]
        )


        if not (
            train_times.max()
            <
            test_times.min()
        ):
            raise (
                MLSplitInvariantError(
                    (
                        "Time holdout violated the "
                        "strict chronological boundary. "
                        "Every training timestamp must "
                        "be earlier than every test "
                        "timestamp."
                    )
                )
            )


        if (
            set(
                train_times.tolist()
            )
            &
            set(
                test_times.tolist()
            )
        ):
            raise (
                MLSplitInvariantError(
                    (
                        "Time holdout produced "
                        "overlapping timestamps across "
                        "train and test."
                    )
                )
            )


    elif isinstance(
        split,
        MLSplitContract,
    ):

        if (
            split.stratify
            and
            not split.shuffle
        ):
            raise (
                MLSplitInputError(
                    (
                        "stratify=True requires "
                        "shuffle=True for the "
                        "Classical ML holdout split."
                    )
                )
            )


        if (
            positional_stratify
            is not None
            and
            not split.stratify
        ):
            raise (
                MLSplitInputError(
                    (
                        "Stratification values were "
                        "provided for a non-stratified "
                        "holdout."
                    )
                )
            )


        random_state = (
            split.random_seed
            if split.shuffle
            else None
        )


        source_positions = (
            np.arange(
                source_row_count,
                dtype=np.int64,
            )
        )


        try:
            (
                train_positions,
                test_positions,
            ) = (
                train_test_split(
                    source_positions,
                    test_size=
                        split.test_size,
                    random_state=
                        random_state,
                    shuffle=
                        split.shuffle,
                    stratify=
                        positional_stratify,
                )
            )

        except ValueError as error:
            raise (
                MLSplitInputError(
                    (
                        "Deterministic train/test "
                        "split could not be created "
                        "from the ML Training Contract."
                    )
                )
            ) from error


    else:
        raise (
            MLSplitInputError(
                "Unsupported ML split contract."
            )
        )


    assert (
        train_positions
        is not None
    )

    assert (
        test_positions
        is not None
    )


    train_positions_tuple = tuple(
        int(
            position
        )
        for position
        in train_positions
    )


    test_positions_tuple = tuple(
        int(
            position
        )
        for position
        in test_positions
    )


    if (
        len(
            train_positions_tuple
        )
        <
        2
        or
        len(
            test_positions_tuple
        )
        <
        2
    ):
        raise (
            MLSplitInputError(
                (
                    "Model Lab v0.1 requires at least "
                    "two training rows and two test "
                    "rows after the holdout split."
                )
            )
        )


    train_set = set(
        train_positions_tuple
    )

    test_set = set(
        test_positions_tuple
    )


    if (
        len(
            train_set
        )
        !=
        len(
            train_positions_tuple
        )
        or
        len(
            test_set
        )
        !=
        len(
            test_positions_tuple
        )
    ):
        raise (
            MLSplitInvariantError(
                (
                    "Shared holdout produced duplicate "
                    "source positions inside one "
                    "partition."
                )
            )
        )


    if (
        train_set
        &
        test_set
    ):
        raise (
            MLSplitInvariantError(
                (
                    "Shared holdout produced overlapping "
                    "train/test source positions."
                )
            )
        )


    valid_positions = set(
        range(
            source_row_count
        )
    )


    observed_positions = (
        train_set
        |
        test_set
    )


    if not observed_positions.issubset(
        valid_positions
    ):
        raise (
            MLSplitInvariantError(
                (
                    "Shared holdout returned a position "
                    "outside the validated source row "
                    "surface."
                )
            )
        )


    purged_positions = tuple(
        sorted(
            valid_positions
            -
            observed_positions
        )
    )


    if (
        not isinstance(
            split,
            MLPurgedGroupTimeHoldoutSplitContract,
        )
        and
        purged_positions
    ):
        raise (
            MLSplitInvariantError(
                (
                    "Non-purged holdout failed to "
                    "partition every validated source "
                    "row."
                )
            )
        )


    return (
        MLHoldoutPartition(
            source_row_count=
                source_row_count,
            train_positions=
                train_positions_tuple,
            test_positions=
                test_positions_tuple,
            purged_positions=
                purged_positions,
        )
    )


def resolve_ml_holdout_partition(
    *,
    x: pd.DataFrame,
    y: pd.Series,
    contract: MLTrainingContract,
    dataframe: pd.DataFrame | None = None,
) -> MLHoldoutPartition:
    """
    Supervised wrapper around the target-free positional
    holdout authority.
    """

    if not isinstance(
        x,
        pd.DataFrame,
    ):
        raise (
            MLSplitInputError(
                "ML holdout features must be a pandas DataFrame."
            )
        )


    if not isinstance(
        y,
        pd.Series,
    ):
        raise (
            MLSplitInputError(
                "ML holdout target must be a pandas Series."
            )
        )


    if (
        len(
            x
        )
        !=
        len(
            y
        )
    ):
        raise (
            MLSplitInputError(
                (
                    "ML holdout feature/target row counts "
                    "must match."
                )
            )
        )


    if not x.index.equals(
        y.index
    ):
        raise (
            MLSplitInputError(
                (
                    "ML holdout feature/target indexes "
                    "must be aligned before positional "
                    "partition resolution."
                )
            )
        )


    split = (
        contract.split
    )


    stratify_values = (
        y
        if (
            isinstance(
                split,
                MLSplitContract,
            )
            and
            contract.problem_type
            ==
            "classification"
            and
            split.stratify
        )
        else None
    )


    partition = (
        resolve_ml_feature_holdout_partition(
            x=
                x,
            contract=
                contract,
            dataframe=
                dataframe,
            stratify_values=
                stratify_values,
        )
    )


    if (
        contract.problem_type
        ==
        "classification"
    ):

        y_train = (
            y.iloc[
                list(
                    partition.train_positions
                )
            ]
        )


        if (
            int(
                y_train.nunique(
                    dropna=False
                )
            )
            <
            2
        ):
            raise (
                MLSplitInputError(
                    (
                        "Classification training split "
                        "contains fewer than two "
                        "classes."
                    )
                )
            )


    return partition
