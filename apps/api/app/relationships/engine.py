from __future__ import annotations

from itertools import (
    combinations,
    product,
)

import re
import unicodedata

import pandas as pd


DIRECT_MATCH_THRESHOLD = 0.80

PARTIAL_MATCH_THRESHOLD = 0.60


RELATIONSHIP_MODE_RANK = {
    "direct": 4,
    "matched_subset": 3,
    "grain_alignment_required": 2,
    "blocked": 1,
}


CARDINALITY_RANK = {
    "1:1": 3,
    "1:N": 2,
    "N:1": 2,
    "N:N": 1,
}


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_column_name(
    column_name: str,
) -> str:
    value = (
        unicodedata
        .normalize(
            "NFKD",
            str(
                column_name
            ),
        )
        .encode(
            "ascii",
            "ignore",
        )
        .decode(
            "ascii",
        )
        .lower()
        .strip()
    )

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    return value.strip(
        "_"
    )


def column_tokens(
    column_name: str,
) -> set[
    str
]:
    normalized = (
        normalize_column_name(
            column_name
        )
    )

    return {
        token
        for token
        in normalized.split(
            "_"
        )
        if token
    }


# ============================================================
# SEMANTIC JOIN ROLES
# ============================================================

def canonical_join_role(
    column_name: str,
) -> str:
    """
    Convert common semantic variants into
    conservative canonical join roles.

    Examples:

    Country
    COUNTRY (DISPLAY)
    pays

        -> country

    REGION (DISPLAY)
    Region

        -> region
    """

    normalized = (
        normalize_column_name(
            column_name
        )
    )

    tokens = (
        column_tokens(
            column_name
        )
    )


    if (
        "country"
        in tokens
        and
        "code"
        in tokens
    ):
        return "country_code"


    if (
        "pays"
        in tokens
        and
        "code"
        in tokens
    ):
        return "country_code"


    if (
        "country"
        in tokens
        or
        "countries"
        in tokens
        or
        "pays"
        in tokens
    ):
        return "country"


    if (
        "region"
        in tokens
        or
        "regions"
        in tokens
    ):
        return "region"


    if (
        "continent"
        in tokens
        or
        "continents"
        in tokens
    ):
        return "continent"


    if (
        "year"
        in tokens
        or
        "annee"
        in tokens
    ):
        return "year"


    if (
        "month"
        in tokens
        or
        "mois"
        in tokens
    ):
        return "month"


    if (
        "date"
        in tokens
        or
        "datetime"
        in tokens
        or
        "timestamp"
        in tokens
    ):
        return "date"


    if (
        "granularity"
        in tokens
        or
        "granularite"
        in tokens
    ):
        return "granularity"


    return normalized


def build_canonical_column_map(
    dataframe: pd.DataFrame,
) -> dict[
    str,
    list[
        str
    ]
]:
    result: dict[
        str,
        list[
            str
        ]
    ] = {}


    for column_name in (
        dataframe.columns
    ):
        actual_name = str(
            column_name
        )

        role = (
            canonical_join_role(
                actual_name
            )
        )


        result.setdefault(
            role,
            [],
        ).append(
            actual_name
        )


    return result


# ============================================================
# VALUE NORMALIZATION
# ============================================================

def normalize_key_value(
    value: object,
) -> str:
    if pd.isna(
        value
    ):
        return ""


    if isinstance(
        value,
        str,
    ):
        return (
            value
            .strip()
            .casefold()
        )


    return str(
        value
    ).strip()


def normalized_unique_values(
    series: pd.Series,
) -> set[
    str
]:
    values: set[
        str
    ] = set()


    for value in (
        series.dropna()
    ):
        normalized = (
            normalize_key_value(
                value
            )
        )

        if normalized:
            values.add(
                normalized
            )


    return values


# ============================================================
# COLUMN-PAIR SELECTION
# ============================================================

def simple_column_overlap(
    left_series: pd.Series,
    right_series: pd.Series,
) -> float:
    left_values = (
        normalized_unique_values(
            left_series
        )
    )

    right_values = (
        normalized_unique_values(
            right_series
        )
    )


    if (
        not left_values
        or
        not right_values
    ):
        return 0.0


    intersection = (
        left_values
        &
        right_values
    )

    union = (
        left_values
        |
        right_values
    )


    if not union:
        return 0.0


    return (
        len(
            intersection
        )
        /
        len(
            union
        )
    )


def choose_column_pair_for_role(
    *,
    left_dataframe: pd.DataFrame,
    right_dataframe: pd.DataFrame,
    left_columns: list[
        str
    ],
    right_columns: list[
        str
    ],
) -> tuple[
    str,
    str,
] | None:
    best_pair: (
        tuple[
            str,
            str,
        ]
        | None
    ) = None

    best_overlap = -1.0


    for (
        left_column,
        right_column,
    ) in product(
        left_columns,
        right_columns,
    ):
        left_unique_count = int(
            left_dataframe[
                left_column
            ]
            .nunique(
                dropna=True
            )
        )

        right_unique_count = int(
            right_dataframe[
                right_column
            ]
            .nunique(
                dropna=True
            )
        )


        if (
            left_unique_count <= 1
            or
            right_unique_count <= 1
        ):
            continue


        overlap = (
            simple_column_overlap(
                left_dataframe[
                    left_column
                ],
                right_dataframe[
                    right_column
                ],
            )
        )


        if (
            overlap >
            best_overlap
        ):
            best_overlap = (
                overlap
            )

            best_pair = (
                left_column,
                right_column,
            )


    return best_pair


# ============================================================
# KEY CONSTRUCTION
# ============================================================

def get_join_key_frame(
    dataframe: pd.DataFrame,
    columns: list[
        str
    ],
) -> pd.DataFrame:
    return (
        dataframe[
            columns
        ]
        .dropna()
        .copy()
    )


def normalized_key_frame(
    dataframe: pd.DataFrame,
    columns: list[
        str
    ],
) -> pd.DataFrame:
    key_frame = (
        get_join_key_frame(
            dataframe,
            columns,
        )
    )


    if key_frame.empty:
        return key_frame


    return key_frame.map(
        normalize_key_value
    )


def key_is_unique(
    dataframe: pd.DataFrame,
    columns: list[
        str
    ],
) -> bool:
    normalized_frame = (
        normalized_key_frame(
            dataframe,
            columns,
        )
    )


    if normalized_frame.empty:
        return False


    return not bool(
        normalized_frame
        .duplicated()
        .any()
    )


def build_key_values(
    dataframe: pd.DataFrame,
    columns: list[
        str
    ],
) -> set[
    tuple[
        str,
        ...
    ]
]:
    normalized_frame = (
        normalized_key_frame(
            dataframe,
            columns,
        )
    )


    if normalized_frame.empty:
        return set()


    return set(
        normalized_frame
        .itertuples(
            index=False,
            name=None,
        )
    )


# ============================================================
# CARDINALITY
# ============================================================

def determine_cardinality(
    left_unique: bool,
    right_unique: bool,
) -> str:
    if (
        left_unique
        and
        right_unique
    ):
        return "1:1"


    if (
        left_unique
        and
        not right_unique
    ):
        return "1:N"


    if (
        not left_unique
        and
        right_unique
    ):
        return "N:1"


    return "N:N"


# ============================================================
# RELATIONSHIP MODE
# ============================================================

def determine_relationship_mode(
    *,
    cardinality: str,
    left_match_rate: float,
    right_match_rate: float,
    overlap_rate: float,
) -> str:
    minimum_match_rate = min(
        left_match_rate,
        right_match_rate,
    )


    if (
        cardinality != "N:N"
        and
        minimum_match_rate >=
        DIRECT_MATCH_THRESHOLD
    ):
        return "direct"


    if (
        cardinality != "N:N"
        and
        minimum_match_rate >=
        PARTIAL_MATCH_THRESHOLD
    ):
        return "matched_subset"


    if (
        cardinality == "N:N"
        and
        overlap_rate >=
        PARTIAL_MATCH_THRESHOLD
    ):
        return (
            "grain_alignment_required"
        )


    return "blocked"


# ============================================================
# RELATIONSHIP SCORE
# ============================================================

def calculate_relationship_score(
    *,
    overlap_rate: float,
    left_match_rate: float,
    right_match_rate: float,
    cardinality: str,
    key_size: int,
    key_roles: list[
        str
    ],
) -> float:
    average_match_rate = (
        left_match_rate
        +
        right_match_rate
    ) / 2


    score = (
        overlap_rate
        *
        35
    )

    score += (
        average_match_rate
        *
        35
    )


    if (
        cardinality ==
        "1:1"
    ):
        score += 20

    elif (
        cardinality
        in {
            "1:N",
            "N:1",
        }
    ):
        score += 15

    else:
        score += 2


    semantic_bonus = 0


    if (
        "country"
        in key_roles
    ):
        semantic_bonus += 4


    if (
        "year"
        in key_roles
    ):
        semantic_bonus += 4


    if (
        "granularity"
        in key_roles
    ):
        semantic_bonus += 2


    if (
        "region"
        in key_roles
    ):
        semantic_bonus += 2


    score += min(
        semantic_bonus,
        8,
    )


    if (
        key_size ==
        1
    ):
        score += 3

    elif (
        key_size ==
        2
    ):
        score += 2

    elif (
        key_size ==
        3
    ):
        score += 1


    return round(
        min(
            score,
            100,
        ),
        2,
    )


# ============================================================
# CANDIDATE ANALYSIS
# ============================================================

def analyze_join_candidate(
    *,
    left_dataframe: pd.DataFrame,
    right_dataframe: pd.DataFrame,
    left_columns: list[
        str
    ],
    right_columns: list[
        str
    ],
    key_roles: list[
        str
    ],
) -> dict[
    str,
    object
]:
    left_unique = (
        key_is_unique(
            left_dataframe,
            left_columns,
        )
    )

    right_unique = (
        key_is_unique(
            right_dataframe,
            right_columns,
        )
    )


    cardinality = (
        determine_cardinality(
            left_unique,
            right_unique,
        )
    )


    left_values = (
        build_key_values(
            left_dataframe,
            left_columns,
        )
    )

    right_values = (
        build_key_values(
            right_dataframe,
            right_columns,
        )
    )


    intersection = (
        left_values
        &
        right_values
    )

    union = (
        left_values
        |
        right_values
    )


    left_match_rate = (
        len(
            intersection
        )
        /
        len(
            left_values
        )
        if left_values
        else
        0.0
    )


    right_match_rate = (
        len(
            intersection
        )
        /
        len(
            right_values
        )
        if right_values
        else
        0.0
    )


    overlap_rate = (
        len(
            intersection
        )
        /
        len(
            union
        )
        if union
        else
        0.0
    )


    relationship_mode = (
        determine_relationship_mode(
            cardinality=
                cardinality,

            left_match_rate=
                left_match_rate,

            right_match_rate=
                right_match_rate,

            overlap_rate=
                overlap_rate,
        )
    )


    score = (
        calculate_relationship_score(
            overlap_rate=
                overlap_rate,

            left_match_rate=
                left_match_rate,

            right_match_rate=
                right_match_rate,

            cardinality=
                cardinality,

            key_size=
                len(
                    left_columns
                ),

            key_roles=
                key_roles,
        )
    )


    warnings: list[
        str
    ] = []


    if (
        cardinality ==
        "N:N"
    ):
        warnings.append(
            (
                "Many-to-many relationship "
                "detected. A direct row-level "
                "join may multiply observations."
            )
        )


    if (
        left_match_rate <
        DIRECT_MATCH_THRESHOLD
    ):
        warnings.append(
            (
                "Some left-side keys have "
                "no matching value on the "
                "right side."
            )
        )


    if (
        right_match_rate <
        DIRECT_MATCH_THRESHOLD
    ):
        warnings.append(
            (
                "Some right-side keys have "
                "no matching value on the "
                "left side."
            )
        )


    if (
        relationship_mode ==
        "matched_subset"
    ):
        warnings.append(
            (
                "The relationship may be "
                "used on the matched subset "
                "if coverage loss is explicitly "
                "reported."
            )
        )


    if (
        relationship_mode ==
        "grain_alignment_required"
    ):
        warnings.append(
            (
                "The datasets appear related, "
                "but their observational grains "
                "must be aligned before a "
                "cross-dataset analysis."
            )
        )


    recommended = (
        relationship_mode ==
        "direct"
    )


    usable_for_analysis = (
        relationship_mode
        in {
            "direct",
            "matched_subset",
        }
    )


    requires_grain_alignment = (
        relationship_mode ==
        "grain_alignment_required"
    )


    return {
        "left_columns":
            left_columns,

        "right_columns":
            right_columns,

        "key_roles":
            key_roles,

        "key_size":
            len(
                left_columns
            ),

        "left_unique":
            left_unique,

        "right_unique":
            right_unique,

        "cardinality":
            cardinality,

        "left_distinct_keys":
            len(
                left_values
            ),

        "right_distinct_keys":
            len(
                right_values
            ),

        "matching_distinct_keys":
            len(
                intersection
            ),

        "left_match_rate":
            round(
                left_match_rate,
                4,
            ),

        "right_match_rate":
            round(
                right_match_rate,
                4,
            ),

        "overlap_rate":
            round(
                overlap_rate,
                4,
            ),

        "relationship_mode":
            relationship_mode,

        "score":
            score,

        "recommended":
            recommended,

        "usable_for_analysis":
            usable_for_analysis,

        "requires_grain_alignment":
            requires_grain_alignment,

        "warnings":
            warnings,
    }


# ============================================================
# CANDIDATE RANKING
# ============================================================

def candidate_sort_key(
    candidate: dict[
        str,
        object
    ],
) -> tuple[
    int,
    int,
    float,
    int,
]:
    relationship_mode = str(
        candidate[
            "relationship_mode"
        ]
    )

    cardinality = str(
        candidate[
            "cardinality"
        ]
    )


    return (
        RELATIONSHIP_MODE_RANK.get(
            relationship_mode,
            0,
        ),

        CARDINALITY_RANK.get(
            cardinality,
            0,
        ),

        float(
            candidate[
                "score"
            ]
        ),

        -int(
            candidate[
                "key_size"
            ]
        ),
    )


# ============================================================
# RELATIONSHIP DISCOVERY
# ============================================================

def discover_relationships(
    datasets: list[
        dict[
            str,
            object
        ]
    ],
) -> list[
    dict[
        str,
        object
    ]
]:
    relationships: list[
        dict[
            str,
            object
        ]
    ] = []


    for (
        left_dataset,
        right_dataset,
    ) in combinations(
        datasets,
        2,
    ):
        left_dataframe = (
            left_dataset[
                "dataframe"
            ]
        )

        right_dataframe = (
            right_dataset[
                "dataframe"
            ]
        )


        if (
            not isinstance(
                left_dataframe,
                pd.DataFrame,
            )
            or
            not isinstance(
                right_dataframe,
                pd.DataFrame,
            )
        ):
            continue


        left_map = (
            build_canonical_column_map(
                left_dataframe
            )
        )

        right_map = (
            build_canonical_column_map(
                right_dataframe
            )
        )


        common_roles = sorted(
            set(
                left_map
            )
            &
            set(
                right_map
            )
        )


        common_column_pairs: list[
            dict[
                str,
                str
            ]
        ] = []


        for role in (
            common_roles
        ):
            selected_pair = (
                choose_column_pair_for_role(
                    left_dataframe=
                        left_dataframe,

                    right_dataframe=
                        right_dataframe,

                    left_columns=
                        left_map[
                            role
                        ],

                    right_columns=
                        right_map[
                            role
                        ],
                )
            )


            if (
                selected_pair
                is None
            ):
                continue


            (
                left_column,
                right_column,
            ) = selected_pair


            common_column_pairs.append(
                {
                    "role":
                        role,

                    "left_column":
                        left_column,

                    "right_column":
                        right_column,
                }
            )


        if (
            not common_column_pairs
        ):
            continue


        candidates: list[
            dict[
                str,
                object
            ]
        ] = []


        max_key_size = min(
            3,
            len(
                common_column_pairs
            ),
        )


        for key_size in range(
            1,
            max_key_size + 1,
        ):
            for candidate_parts in (
                combinations(
                    common_column_pairs,
                    key_size,
                )
            ):
                left_columns = [
                    part[
                        "left_column"
                    ]
                    for part
                    in candidate_parts
                ]

                right_columns = [
                    part[
                        "right_column"
                    ]
                    for part
                    in candidate_parts
                ]

                key_roles = [
                    part[
                        "role"
                    ]
                    for part
                    in candidate_parts
                ]


                candidate = (
                    analyze_join_candidate(
                        left_dataframe=
                            left_dataframe,

                        right_dataframe=
                            right_dataframe,

                        left_columns=
                            left_columns,

                        right_columns=
                            right_columns,

                        key_roles=
                            key_roles,
                    )
                )


                candidates.append(
                    candidate
                )


        candidates.sort(
            key=
                candidate_sort_key,

            reverse=True,
        )


        best_candidate = (
            candidates[
                0
            ]
            if candidates
            else
            None
        )


        relationships.append(
            {
                "left_dataset":
                    {
                        "dataset_id":
                            left_dataset[
                                "dataset_id"
                            ],

                        "filename":
                            left_dataset[
                                "filename"
                            ],
                    },

                "right_dataset":
                    {
                        "dataset_id":
                            right_dataset[
                                "dataset_id"
                            ],

                        "filename":
                            right_dataset[
                                "filename"
                            ],
                    },

                "common_columns":
                    [
                        {
                            "role":
                                item[
                                    "role"
                                ],

                            "left":
                                item[
                                    "left_column"
                                ],

                            "right":
                                item[
                                    "right_column"
                                ],
                        }
                        for item
                        in common_column_pairs
                    ],

                "best_candidate":
                    best_candidate,

                "candidates":
                    candidates,
            }
        )


    relationships.sort(
        key=lambda relationship:
            (
                candidate_sort_key(
                    relationship[
                        "best_candidate"
                    ]
                )
                if (
                    relationship[
                        "best_candidate"
                    ]
                    is not None
                )
                else (
                    0,
                    0,
                    0.0,
                    0,
                )
            ),
        reverse=True,
    )


    return relationships