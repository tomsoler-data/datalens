from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

from app.evals.dataset_dependency_contract_v0_8 import (
    DatasetDependencyCandidate,
)


# ============================================================
# VERSION
# ============================================================

DATASET_DEPENDENCY_SCORER_VERSION = (
    "dataset_dependency_scorer_v0.8"
)


# ============================================================
# NUMERIC PRECISION
# ============================================================

SCORE_PRECISION = 12


def _score(
    value: float,
) -> float:
    """
    Normalize floating-point evaluation metrics.

    Evaluation scores are semantic values in [0, 1].
    Small IEEE-754 representation artifacts such as:

        0.9999999999999999

    should be persisted as:

        1.0

    This keeps deterministic evaluation outputs stable and
    allows exact assertions for mathematically exact scores.
    """

    normalized = round(
        float(
            value
        ),
        SCORE_PRECISION,
    )


    if normalized < 0.0:
        return 0.0


    if normalized > 1.0:
        return 1.0


    return normalized


# ============================================================
# TYPES
# ============================================================

DatasetGroup = tuple[
    str,
    ...
]


DatasetPair = tuple[
    str,
    str,
]


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize_dataset_id(
    value: str,
) -> str:
    """
    Dataset IDs are structural identifiers.

    We trim accidental whitespace but otherwise preserve the
    identifier semantics.
    """

    return value.strip()


def _canonical_group(
    dataset_ids: list[str],
) -> DatasetGroup:

    normalized = [
        _normalize_dataset_id(
            dataset_id
        )

        for dataset_id
        in dataset_ids
    ]


    return tuple(
        sorted(
            normalized
        )
    )


def _candidate_groups(
    candidate: DatasetDependencyCandidate,
) -> list[
    DatasetGroup
]:

    return [
        _canonical_group(
            requirement.dataset_ids
        )

        for requirement
        in candidate.requirements
    ]


def _expected_groups(
    expected_groups: list[
        list[str]
    ],
) -> list[
    DatasetGroup
]:

    return [
        _canonical_group(
            dataset_ids
        )

        for dataset_ids
        in expected_groups
    ]


# ============================================================
# VALIDATE EXPECTATION
# ============================================================

def _validate_expected_groups(
    *,
    expected_groups: list[
        list[str]
    ],
    allowed_dataset_ids: set[str],
) -> None:

    if not expected_groups:
        raise ValueError(
            "expected_groups must contain at least "
            "one analytical requirement."
        )


    canonical_groups: list[
        DatasetGroup
    ] = []


    for group in expected_groups:

        if not group:
            raise ValueError(
                "Expected dependency groups "
                "must not be empty."
            )


        normalized_group = [
            _normalize_dataset_id(
                dataset_id
            )

            for dataset_id
            in group
        ]


        if any(
            not dataset_id
            for dataset_id
            in normalized_group
        ):
            raise ValueError(
                "Expected dataset IDs "
                "must not be empty."
            )


        if (
            len(
                normalized_group
            )
            != len(
                set(
                    normalized_group
                )
            )
        ):
            raise ValueError(
                "Expected dependency groups must not "
                "contain duplicate dataset IDs."
            )


        unknown = (
            set(
                normalized_group
            )
            - allowed_dataset_ids
        )


        if unknown:
            raise ValueError(
                "Expected dependency group references "
                "unknown dataset(s): "
                f"{sorted(unknown)}"
            )


        canonical_groups.append(
            tuple(
                sorted(
                    normalized_group
                )
            )
        )


    if (
        len(
            canonical_groups
        )
        != len(
            set(
                canonical_groups
            )
        )
    ):
        raise ValueError(
            "Expected dependency groups "
            "must not contain duplicates."
        )


# ============================================================
# SET F1
# ============================================================

def _set_f1(
    *,
    expected: set[str],
    actual: set[str],
) -> float:

    if (
        not expected
        and not actual
    ):
        return 1.0


    if (
        not expected
        or not actual
    ):
        return 0.0


    overlap = len(
        expected
        & actual
    )


    precision = (
        overlap
        / len(
            actual
        )
    )


    recall = (
        overlap
        / len(
            expected
        )
    )


    if (
        precision
        + recall
        == 0
    ):
        return 0.0


    return _score(
        (
            2
            * precision
            * recall
        )
        / (
            precision
            + recall
        )
    )


# ============================================================
# FLAT DATASET SET
# ============================================================

def _flatten_groups(
    groups: list[
        DatasetGroup
    ],
) -> set[str]:

    return {
        dataset_id

        for group
        in groups

        for dataset_id
        in group
    }


# ============================================================
# CO-REQUIREMENT PAIRS
# ============================================================

def _co_requirement_pairs(
    groups: list[
        DatasetGroup
    ],
) -> set[
    DatasetPair
]:
    """
    Represent grouping structure using unordered dataset pairs.

    Examples
    --------

    [["sales", "support"]]

        -> {("sales", "support")}

    [["sales"], ["support"]]

        -> {}

    This catches the critical difference between datasets
    required together and datasets required independently.
    """

    pairs: set[
        DatasetPair
    ] = set()


    for group in groups:

        for (
            left,
            right,
        ) in combinations(
            group,
            2,
        ):

            pairs.add(
                tuple(
                    sorted(
                        (
                            left,
                            right,
                        )
                    )
                )
            )


    return pairs


def _pairwise_f1(
    *,
    expected_groups: list[
        DatasetGroup
    ],
    actual_groups: list[
        DatasetGroup
    ],
) -> float:

    expected_pairs = (
        _co_requirement_pairs(
            expected_groups
        )
    )


    actual_pairs = (
        _co_requirement_pairs(
            actual_groups
        )
    )


    if (
        not expected_pairs
        and not actual_pairs
    ):
        return 1.0


    if (
        not expected_pairs
        or not actual_pairs
    ):
        return 0.0


    overlap = len(
        expected_pairs
        & actual_pairs
    )


    precision = (
        overlap
        / len(
            actual_pairs
        )
    )


    recall = (
        overlap
        / len(
            expected_pairs
        )
    )


    if (
        precision
        + recall
        == 0
    ):
        return 0.0


    return _score(
        (
            2
            * precision
            * recall
        )
        / (
            precision
            + recall
        )
    )


# ============================================================
# RESULT
# ============================================================

@dataclass(
    frozen=True,
)
class DatasetDependencyScoreV08:

    scorer_version: str

    exact_groups: float

    dataset_f1: float

    pairwise_grouping_f1: float

    requirement_count: float

    expected_groups: tuple[
        DatasetGroup,
        ...
    ]

    actual_groups: tuple[
        DatasetGroup,
        ...
    ]

    missing_dataset_ids: tuple[
        str,
        ...
    ]

    hallucinated_dataset_ids: tuple[
        str,
        ...
    ]

    duplicate_requirement_groups: tuple[
        DatasetGroup,
        ...
    ]

    overall: float


    def as_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:

        return {
            "scorer_version":
                self.scorer_version,

            "metrics": {
                "exact_groups":
                    self.exact_groups,

                "dataset_f1":
                    self.dataset_f1,

                "pairwise_grouping_f1":
                    self.pairwise_grouping_f1,

                "requirement_count":
                    self.requirement_count,
            },

            "diagnostics": {
                "expected_groups": [
                    list(
                        group
                    )

                    for group
                    in self.expected_groups
                ],

                "actual_groups": [
                    list(
                        group
                    )

                    for group
                    in self.actual_groups
                ],

                "missing_dataset_ids":
                    list(
                        self.missing_dataset_ids
                    ),

                "hallucinated_dataset_ids":
                    list(
                        self.hallucinated_dataset_ids
                    ),

                "duplicate_requirement_groups": [
                    list(
                        group
                    )

                    for group
                    in self.duplicate_requirement_groups
                ],
            },

            "overall":
                self.overall,
        }


# ============================================================
# SCORER
# ============================================================

def score_dataset_dependency_candidate(
    *,
    candidate: DatasetDependencyCandidate,
    expected_groups: list[
        list[str]
    ],
    allowed_dataset_ids: set[str],
) -> DatasetDependencyScoreV08:

    normalized_allowed = {
        _normalize_dataset_id(
            dataset_id
        )

        for dataset_id
        in allowed_dataset_ids
    }


    _validate_expected_groups(
        expected_groups=(
            expected_groups
        ),

        allowed_dataset_ids=(
            normalized_allowed
        ),
    )


    expected = (
        _expected_groups(
            expected_groups
        )
    )


    actual = (
        _candidate_groups(
            candidate
        )
    )


    # ========================================================
    # FLAT DATASET REFERENCES
    # ========================================================

    expected_dataset_ids = (
        _flatten_groups(
            expected
        )
    )


    actual_dataset_ids = (
        _flatten_groups(
            actual
        )
    )


    missing_dataset_ids = tuple(
        sorted(
            expected_dataset_ids
            - actual_dataset_ids
        )
    )


    hallucinated_dataset_ids = tuple(
        sorted(
            actual_dataset_ids
            - normalized_allowed
        )
    )


    # ========================================================
    # DUPLICATE REQUIREMENT GROUPS
    # ========================================================

    seen_groups: set[
        DatasetGroup
    ] = set()


    duplicate_groups: set[
        DatasetGroup
    ] = set()


    for group in actual:

        if group in seen_groups:
            duplicate_groups.add(
                group
            )

        else:
            seen_groups.add(
                group
            )


    duplicate_requirement_groups = tuple(
        sorted(
            duplicate_groups
        )
    )


    # ========================================================
    # EXACT GROUP STRUCTURE
    #
    # Requirement IDs and ordering are irrelevant.
    #
    # Duplicate requirements are NOT accepted as exact.
    # ========================================================

    expected_group_set = set(
        expected
    )


    actual_group_set = set(
        actual
    )


    exact_groups_score = (
        1.0

        if (
            expected_group_set
            == actual_group_set

            and not duplicate_requirement_groups

            and len(
                expected
            )
            == len(
                actual
            )
        )

        else 0.0
    )


    # ========================================================
    # DATASET COVERAGE
    # ========================================================

    dataset_f1 = (
        _set_f1(
            expected=(
                expected_dataset_ids
            ),

            actual=(
                actual_dataset_ids
            ),
        )
    )


    # ========================================================
    # GROUPING STRUCTURE
    # ========================================================

    pairwise_grouping_f1 = (
        _pairwise_f1(
            expected_groups=(
                expected
            ),

            actual_groups=(
                actual
            ),
        )
    )


    # ========================================================
    # REQUIREMENT COUNT
    # ========================================================

    requirement_count_score = (
        1.0

        if (
            len(
                expected
            )
            == len(
                actual
            )
        )

        else 0.0
    )


    # ========================================================
    # OVERALL
    #
    # Exact grouping is the most important property.
    #
    # 50% exact groups
    # 20% dataset coverage
    # 20% grouping relationships
    # 10% number of analytical requirements
    #
    # IMPORTANT:
    #
    # The final result is normalized through _score() to avoid
    # IEEE-754 artifacts such as 0.9999999999999999.
    # ========================================================

    overall = _score(
        exact_groups_score
        * 0.50

        + dataset_f1
        * 0.20

        + pairwise_grouping_f1
        * 0.20

        + requirement_count_score
        * 0.10
    )


    return DatasetDependencyScoreV08(
        scorer_version=(
            DATASET_DEPENDENCY_SCORER_VERSION
        ),

        exact_groups=(
            _score(
                exact_groups_score
            )
        ),

        dataset_f1=(
            _score(
                dataset_f1
            )
        ),

        pairwise_grouping_f1=(
            _score(
                pairwise_grouping_f1
            )
        ),

        requirement_count=(
            _score(
                requirement_count_score
            )
        ),

        expected_groups=tuple(
            sorted(
                expected
            )
        ),

        actual_groups=tuple(
            sorted(
                actual
            )
        ),

        missing_dataset_ids=(
            missing_dataset_ids
        ),

        hallucinated_dataset_ids=(
            hallucinated_dataset_ids
        ),

        duplicate_requirement_groups=(
            duplicate_requirement_groups
        ),

        overall=(
            overall
        ),
    )