from __future__ import annotations


import re
import unicodedata

from typing import (
    Any,
)


import pandas as pd


from app.discovery.schemas import (
    AnalysisDiscoveryReport,
    DiscoveredAnalysis,
    DiscoveredVariable,
)

from app.execution.single_schemas import (
    SingleDatasetExecutedAnalysis,
    SingleDatasetExecutionReport,
)

from app.profiling.types import (
    infer_analytical_type,
)


# ============================================================
# VERSION
# ============================================================

DERIVED_DISCOVERY_POLICY_VERSION = (
    "derived_discovery_policy_v0.3"
)


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(
    value: object,
) -> str:
    text = (
        unicodedata
        .normalize(
            "NFKD",
            str(
                value
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


    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )


    return text.strip(
        "_"
    )


def text_tokens(
    value: object,
) -> set[str]:
    return {
        token

        for token
        in normalize_text(
            value
        ).split(
            "_"
        )

        if token
    }


def native_scalar(
    value: Any,
) -> Any:
    if hasattr(
        value,
        "item",
    ):
        try:
            value = value.item()

        except (
            ValueError,
            AttributeError,
        ):
            pass


    try:
        if pd.isna(
            value
        ):
            return None

    except (
        TypeError,
        ValueError,
    ):
        pass


    return value


# ============================================================
# OBJECTIVE HELPERS
# ============================================================

def objective_group_bonus(
    objective: str | None,
    group_column: str,
) -> float:
    if not objective:
        return 0.0


    objective_normalized = (
        normalize_text(
            objective
        )
    )


    objective_tokens = (
        text_tokens(
            objective
        )
    )


    group_normalized = (
        normalize_text(
            group_column
        )
    )


    group_tokens = (
        text_tokens(
            group_column
        )
    )


    if (
        objective_tokens
        &
        group_tokens
    ):
        return 15.0


    aliases: dict[
        str,
        set[str],
    ] = {
        "category": {
            "categ",
            "category",
            "categorie",
            "categories",
        },

        "sex": {
            "sex",
            "sexe",
            "gender",
            "genre",
        },

        "region": {
            "region",
            "regions",
            "continent",
        },

        "country": {
            "country",
            "countries",
            "pays",
        },

        "channel": {
            "channel",
            "canal",
            "channels",
            "canaux",
        },
    }


    for signals in (
        aliases.values()
    ):
        group_matches = any(
            signal
            in group_normalized

            for signal
            in signals
        )


        objective_matches = any(
            signal
            in objective_normalized

            for signal
            in signals
        )


        if (
            group_matches
            and
            objective_matches
        ):
            return 15.0


    return 0.0


def objective_measure_bonus(
    objective: str | None,
    measure_column: str,
) -> float:
    if not objective:
        return 0.0


    objective_normalized = (
        normalize_text(
            objective
        )
    )


    measure_normalized = (
        normalize_text(
            measure_column
        )
    )


    monetary_objective_signals = {
        "chiffre_d_affaires",
        "chiffre_affaires",
        "ca",
        "revenu",
        "revenue",
        "ventes",
        "vente",
        "sales",
        "turnover",
    }


    monetary_measure_signals = {
        "price",
        "prix",
        "amount",
        "montant",
        "revenue",
        "revenu",
        "sales",
        "turnover",
        "spend",
        "cost",
        "cout",
    }


    objective_is_monetary = any(
        signal
        in objective_normalized

        for signal
        in monetary_objective_signals
    )


    measure_is_monetary = any(
        signal
        in measure_normalized

        for signal
        in monetary_measure_signals
    )


    if (
        objective_is_monetary
        and
        measure_is_monetary
    ):
        return 8.0


    return 0.0


# ============================================================
# DATASET LOOKUP
# ============================================================

def build_dataset_map(
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:
    return {
        str(
            dataset[
                "dataset_id"
            ]
        ):
            dataset

        for dataset
        in datasets
    }


# ============================================================
# CANDIDATE HELPERS
# ============================================================

def candidate_variable(
    candidate: DiscoveredAnalysis,
    *,
    role: str,
) -> DiscoveredVariable | None:
    for variable in (
        candidate.variables
    ):
        if (
            variable.role
            ==
            role
        ):
            return variable


    return None


def candidate_columns(
    candidate: DiscoveredAnalysis,
) -> set[str]:
    return {
        variable.column

        for variable
        in candidate.variables
    }


# ============================================================
# FUNCTIONAL DEPENDENCY
# ============================================================

def column_is_functionally_dependent(
    dataframe: pd.DataFrame,
    *,
    determinant: str,
    dependent: str,
) -> bool:
    if (
        determinant
        not in dataframe.columns
        or
        dependent
        not in dataframe.columns
    ):
        return False


    working = (
        dataframe[
            [
                determinant,
                dependent,
            ]
        ]
        .dropna(
            subset=[
                determinant
            ]
        )
    )


    if working.empty:
        return False


    counts = (
        working
        .groupby(
            determinant,
            dropna=False,
        )[
            dependent
        ]
        .nunique(
            dropna=True
        )
    )


    if counts.empty:
        return False


    return bool(
        (
            counts
            <=
            1
        )
        .all()
    )


# ============================================================
# POLICY RULE:
# FALSE DERIVED TIMELINES
# ============================================================

def should_suppress_derived_time_series(
    candidate: DiscoveredAnalysis,
    dataset: dict[
        str,
        Any,
    ],
) -> bool:
    if (
        candidate.family
        !=
        "time_series"
    ):
        return False


    derivation_type = str(
        dataset.get(
            "derivation_type",
            "",
        )
    )


    return (
        derivation_type
        !=
        "monthly_additive_measure"
    )


# ============================================================
# POLICY RULE:
# STRUCTURAL ASSOCIATIONS
# ============================================================

def should_suppress_structural_association(
    candidate: DiscoveredAnalysis,
    dataset: dict[
        str,
        Any,
    ],
) -> bool:
    if (
        candidate.family
        !=
        "quantitative_association"
    ):
        return False


    provenance = (
        dataset.get(
            "provenance",
            {}
        )
    )


    target_measure = str(
        provenance.get(
            "target_measure_column",
            "",
        )
    )


    if not target_measure:
        return False


    columns = (
        candidate_columns(
            candidate
        )
    )


    structural_pair = {
        target_measure,
        "event_count",
    }


    return (
        columns
        ==
        structural_pair
    )


# ============================================================
# POLICY RULE:
# REPEATED-PARENT QUANTITATIVE ASSOCIATION
# ============================================================

def should_suppress_repeated_parent_association(
    candidate: DiscoveredAnalysis,
    dataset: dict[
        str,
        Any,
    ],
) -> bool:
    """
    Suppress naive quantitative associations at a
    child grain when observations repeatedly belong
    to the same higher-level parent entity.

    Example:

        one row per session
            ↓
        many sessions per client

    A simple Spearman correlation over sessions
    would treat 345k sessions as independent even
    though they belong to a much smaller set of
    customers.

    Until DataLens has a repeated-measures /
    hierarchical method for this situation, such
    candidates are not promoted to execution.
    """

    if (
        candidate.family
        !=
        "quantitative_association"
    ):
        return False


    provenance = (
        dataset.get(
            "provenance",
            {}
        )
    )


    parent_column = (
        provenance.get(
            "parent_entity_column"
        )
    )


    if not parent_column:
        return False


    dataframe = dataset.get(
        "dataframe"
    )


    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        return False


    parent_column = str(
        parent_column
    )


    if (
        parent_column
        not in dataframe.columns
    ):
        return False


    valid_parent = (
        dataframe[
            parent_column
        ]
        .dropna()
    )


    if (
        len(
            valid_parent
        )
        <
        2
    ):
        return False


    parent_count = int(
        valid_parent
        .nunique()
    )


    row_count = int(
        len(
            valid_parent
        )
    )


    return (
        parent_count
        <
        row_count
    )


# ============================================================
# POLICY RULE:
# REPEATED PARENT GROUPING
# ============================================================

def should_suppress_repeated_parent_grouping(
    candidate: DiscoveredAnalysis,
    dataset: dict[
        str,
        Any,
    ],
) -> bool:
    if (
        candidate.family
        !=
        "group_comparison"
    ):
        return False


    if (
        dataset.get(
            "derivation_type"
        )
        !=
        "entity_additive_measure"
    ):
        return False


    provenance = (
        dataset.get(
            "provenance",
            {}
        )
    )


    grain_column = str(
        provenance.get(
            "entity_column",
            "",
        )
    )


    dataframe = dataset.get(
        "dataframe"
    )


    if (
        not grain_column
        or
        not isinstance(
            dataframe,
            pd.DataFrame,
        )
    ):
        return False


    group_variable = (
        candidate_variable(
            candidate,
            role=
                "group",
        )
    )


    if (
        group_variable
        is None
    ):
        return False


    group_column = (
        group_variable.column
    )


    if (
        group_column
        not in dataframe.columns
    ):
        return False


    for column in (
        dataframe.columns
    ):
        column_name = str(
            column
        )


        if (
            column_name
            ==
            grain_column
        ):
            continue


        inferred = (
            infer_analytical_type(
                column_name,
                dataframe[
                    column_name
                ],
            )
        )


        if (
            inferred.get(
                "type"
            )
            !=
            "identifier"
        ):
            continue


        valid = (
            dataframe[
                column_name
            ]
            .dropna()
        )


        if valid.empty:
            continue


        if (
            valid.nunique()
            >=
            len(
                valid
            )
        ):
            continue


        if column_is_functionally_dependent(
            dataframe,
            determinant=
                column_name,
            dependent=
                group_column,
        ):
            return True


    return False


# ============================================================
# AGGREGATE COMPOSITION
# ============================================================

def calculate_aggregate_composition(
    dataframe: pd.DataFrame,
    *,
    group_column: str,
    measure_column: str,
) -> dict[
    str,
    Any,
] | None:
    working = (
        dataframe[
            [
                group_column,
                measure_column,
            ]
        ]
        .copy()
    )


    working[
        measure_column
    ] = pd.to_numeric(
        working[
            measure_column
        ],
        errors="coerce",
    )


    working = (
        working
        .dropna(
            subset=[
                group_column,
                measure_column,
            ]
        )
    )


    if (
        len(
            working
        )
        <
        2
    ):
        return None


    if (
        working[
            group_column
        ]
        .duplicated()
        .any()
    ):
        return None


    total = float(
        working[
            measure_column
        ]
        .sum()
    )


    if (
        abs(
            total
        )
        <=
        1e-12
    ):
        return None


    ordered = (
        working
        .sort_values(
            measure_column,
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []


    shares: list[
        float
    ] = []


    for index, row in (
        ordered.iterrows()
    ):
        value = float(
            row[
                measure_column
            ]
        )


        share = (
            value
            /
            total
        )


        shares.append(
            share
        )


        rows.append(
            {
                "group":
                    native_scalar(
                        row[
                            group_column
                        ]
                    ),

                "rank":
                    index
                    +
                    1,

                "value":
                    value,

                "share":
                    share,
            }
        )


    return {
        "total_value":
            total,

        "group_count":
            len(
                rows
            ),

        "top_group":
            rows[
                0
            ][
                "group"
            ],

        "top_value":
            rows[
                0
            ][
                "value"
            ],

        "top_share":
            rows[
                0
            ][
                "share"
            ],

        "min_share":
            min(
                shares
            ),

        "max_share":
            max(
                shares
            ),

        "share_spread":
            max(
                shares
            )
            -
            min(
                shares
            ),

        "rows":
            rows,
    }


# ============================================================
# AGGREGATE BREAKDOWN CANDIDATE
# ============================================================

def build_aggregate_breakdown_candidate(
    dataset: dict[
        str,
        Any,
    ],
    *,
    objective: str | None,
) -> DiscoveredAnalysis | None:
    if (
        dataset.get(
            "derivation_type"
        )
        !=
        "categorical_additive_measure"
    ):
        return None


    dataframe = dataset.get(
        "dataframe"
    )


    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        return None


    provenance = (
        dataset.get(
            "provenance",
            {}
        )
    )


    group_column = str(
        provenance.get(
            "group_column",
            "",
        )
    )


    measure_column = str(
        provenance.get(
            "target_measure_column",
            "",
        )
    )


    if (
        not group_column
        or
        not measure_column
        or
        group_column
        not in dataframe.columns
        or
        measure_column
        not in dataframe.columns
    ):
        return None


    composition = (
        calculate_aggregate_composition(
            dataframe,
            group_column=
                group_column,
            measure_column=
                measure_column,
        )
    )


    if composition is None:
        return None


    source_measure = str(
        provenance.get(
            "source_measure_column",
            measure_column,
        )
    )


    group_bonus = (
        objective_group_bonus(
            objective,
            group_column,
        )
    )


    measure_bonus = (
        objective_measure_bonus(
            objective,
            source_measure,
        )
    )


    priority = (
        70.0
        +
        group_bonus
        +
        measure_bonus
    )


    priority = max(
        0.0,
        min(
            100.0,
            priority,
        ),
    )


    dataset_id = str(
        dataset[
            "dataset_id"
        ]
    )


    filename = str(
        dataset[
            "filename"
        ]
    )


    group_type = (
        infer_analytical_type(
            group_column,
            dataframe[
                group_column
            ],
        )
    )


    measure_type = (
        infer_analytical_type(
            measure_column,
            dataframe[
                measure_column
            ],
        )
    )


    return DiscoveredAnalysis(
        analysis_id=(
            f"{dataset_id}:"
            "aggregate_breakdown:"
            f"{normalize_text(group_column)}:"
            f"{normalize_text(measure_column)}"
        ),

        scope=
            "single_dataset",

        family=
            "aggregate_breakdown",

        title=(
            f"Répartition de "
            f"{measure_column} par "
            f"{group_column}"
        ),

        priority_score=
            round(
                priority,
                2,
            ),

        readiness=
            "executable_now",

        datasets=[
            filename
        ],

        dataset_ids=[
            dataset_id
        ],

        variables=[
            DiscoveredVariable(
                dataset_id=
                    dataset_id,

                dataset_filename=
                    filename,

                column=
                    group_column,

                role=
                    "group",

                analysis_kind=
                    str(
                        group_type.get(
                            "type",
                            "categorical",
                        )
                    ),

                semantic_role=
                    "category",

                concepts=[],
            ),

            DiscoveredVariable(
                dataset_id=
                    dataset_id,

                dataset_filename=
                    filename,

                column=
                    measure_column,

                role=
                    "value",

                analysis_kind=
                    str(
                        measure_type.get(
                            "type",
                            "quantitative",
                        )
                    ),

                semantic_role=
                    "measure",

                concepts=[],
            ),
        ],

        chart_type=
            "bar",

        execution_strategy=
            "deterministic_aggregate_breakdown",

        why_interesting=[
            (
                "Le dataset représente déjà un "
                "grain catégoriel agrégé."
            ),

            (
                "La mesure additive peut être "
                "décomposée directement entre "
                "les catégories sans test "
                "inférentiel."
            ),
        ],

        limitations=[
            (
                "Cette analyse décrit la "
                "répartition du total observé ; "
                "elle ne constitue pas un test "
                "statistique entre échantillons."
            )
        ],

        observed_signals={
            "semantic_family":
                "aggregate_breakdown",

            "group_count":
                composition[
                    "group_count"
                ],

            "source_measure_column":
                source_measure,

            "aggregation":
                provenance.get(
                    "aggregation"
                ),

            "grain":
                provenance.get(
                    "grain"
                ),

            "objective_group_bonus":
                group_bonus,

            "objective_measure_bonus":
                measure_bonus,

            "total_value":
                composition[
                    "total_value"
                ],

            "top_group":
                composition[
                    "top_group"
                ],

            "top_share":
                composition[
                    "top_share"
                ],

            "min_share":
                composition[
                    "min_share"
                ],

            "max_share":
                composition[
                    "max_share"
                ],

            "share_spread":
                composition[
                    "share_spread"
                ],

            "provenance":
                provenance,

            "derived_policy_version":
                DERIVED_DISCOVERY_POLICY_VERSION,
        },

        redundancy_key=(
            "aggregate_breakdown:"
            f"{dataset_id}:"
            f"{group_column}:"
            f"{measure_column}"
        ),
    )


# ============================================================
# DISCOVERY POLICY
# ============================================================

def apply_derived_discovery_policy(
    discovery: AnalysisDiscoveryReport,
    *,
    derived_datasets: list[
        dict[
            str,
            Any,
        ]
    ],
    objective: str | None = None,
) -> AnalysisDiscoveryReport:
    dataset_map = (
        build_dataset_map(
            derived_datasets
        )
    )


    retained: list[
        DiscoveredAnalysis
    ] = []


    suppression_counts = {
        "false_time_series":
            0,

        "structural_association":
            0,

        "repeated_parent_association":
            0,

        "repeated_parent_grouping":
            0,

        "derived_quality":
            0,
    }


    for candidate in (
        discovery.candidates
    ):
        if (
            candidate.scope
            !=
            "single_dataset"
        ):
            continue


        if (
            len(
                candidate.dataset_ids
            )
            !=
            1
        ):
            continue


        dataset_id = str(
            candidate.dataset_ids[
                0
            ]
        )


        dataset = (
            dataset_map.get(
                dataset_id
            )
        )


        if dataset is None:
            continue


        if (
            candidate.family
            ==
            "data_quality"
        ):
            suppression_counts[
                "derived_quality"
            ] += 1

            continue


        if should_suppress_derived_time_series(
            candidate,
            dataset,
        ):
            suppression_counts[
                "false_time_series"
            ] += 1

            continue


        if should_suppress_structural_association(
            candidate,
            dataset,
        ):
            suppression_counts[
                "structural_association"
            ] += 1

            continue


        if should_suppress_repeated_parent_association(
            candidate,
            dataset,
        ):
            suppression_counts[
                "repeated_parent_association"
            ] += 1

            continue


        if should_suppress_repeated_parent_grouping(
            candidate,
            dataset,
        ):
            suppression_counts[
                "repeated_parent_grouping"
            ] += 1

            continue


        candidate.observed_signals[
            "derived_discovery_policy"
        ] = {
            "status":
                "retained",

            "version":
                DERIVED_DISCOVERY_POLICY_VERSION,
        }


        retained.append(
            candidate
        )


    breakdown_candidates: list[
        DiscoveredAnalysis
    ] = []


    for dataset in (
        derived_datasets
    ):
        candidate = (
            build_aggregate_breakdown_candidate(
                dataset,
                objective=
                    objective,
            )
        )


        if candidate is not None:
            breakdown_candidates.append(
                candidate
            )


    selected: dict[
        str,
        DiscoveredAnalysis,
    ] = {}


    for candidate in [
        *retained,
        *breakdown_candidates,
    ]:
        existing = (
            selected.get(
                candidate.redundancy_key
            )
        )


        if (
            existing is None
            or
            candidate.priority_score
            >
            existing.priority_score
        ):
            selected[
                candidate.redundancy_key
            ] = candidate


    final_candidates = list(
        selected.values()
    )


    final_candidates.sort(
        key=lambda candidate:
            candidate.priority_score,
        reverse=True,
    )


    discovery.candidates = (
        final_candidates
    )


    discovery.candidate_count = (
        len(
            final_candidates
        )
    )


    discovery.single_dataset_candidate_count = (
        len(
            final_candidates
        )
    )


    discovery.cross_dataset_candidate_count = (
        0
    )


    discovery.discovery_notes.extend(
        [
            (
                "Derived Discovery Policy "
                f"{DERIVED_DISCOVERY_POLICY_VERSION} "
                "was applied."
            ),

            (
                "Derived entity-grain datasets "
                "are not treated as time series "
                "merely because they contain a "
                "birth-year attribute."
            ),

            (
                "Associations between an additive "
                "measure and event_count produced "
                "by the same GROUP BY are treated "
                "as structural and suppressed."
            ),

            (
                "Quantitative associations at a "
                "child grain are suppressed when "
                "rows repeatedly belong to the same "
                "parent entity and no specialized "
                "repeated-measures method is "
                "available."
            ),

            (
                "Group comparisons at a child "
                "grain are suppressed when the "
                "group attribute belongs to a "
                "repeated parent entity."
            ),

            (
                f"{len(breakdown_candidates)} "
                "aggregate breakdown candidate(s) "
                "were added."
            ),

            (
                "Suppressed derived candidates — "
                f"time series: "
                f"{suppression_counts['false_time_series']}; "
                f"structural associations: "
                f"{suppression_counts['structural_association']}; "
                f"repeated-parent associations: "
                f"{suppression_counts['repeated_parent_association']}; "
                f"repeated-parent comparisons: "
                f"{suppression_counts['repeated_parent_grouping']}; "
                f"quality audits: "
                f"{suppression_counts['derived_quality']}."
            ),
        ]
    )


    return discovery


# ============================================================
# AGGREGATE BREAKDOWN EXECUTION
# ============================================================

def execute_aggregate_breakdown(
    candidate: DiscoveredAnalysis,
    *,
    dataset: dict[
        str,
        Any,
    ],
) -> SingleDatasetExecutedAnalysis:
    dataframe = dataset[
        "dataframe"
    ]


    dataset_id = str(
        dataset[
            "dataset_id"
        ]
    )


    dataset_name = str(
        dataset[
            "filename"
        ]
    )


    group_variable = (
        candidate_variable(
            candidate,
            role=
                "group",
        )
    )


    value_variable = (
        candidate_variable(
            candidate,
            role=
                "value",
        )
    )


    if (
        group_variable is None
        or
        value_variable is None
    ):
        return SingleDatasetExecutedAnalysis(
            analysis_id=
                candidate.analysis_id,

            title=
                candidate.title,

            family=
                "group_comparison",

            dataset_id=
                dataset_id,

            dataset=
                dataset_name,

            execution_status=
                "failed",

            variables=[],

            warnings=[
                (
                    "Aggregate breakdown variables "
                    "are unavailable."
                )
            ],

            limitations=
                candidate.limitations,

            execution_rule_version=
                DERIVED_DISCOVERY_POLICY_VERSION,
        )


    group_column = (
        group_variable.column
    )


    measure_column = (
        value_variable.column
    )


    if (
        group_column
        not in dataframe.columns
        or
        measure_column
        not in dataframe.columns
    ):
        return SingleDatasetExecutedAnalysis(
            analysis_id=
                candidate.analysis_id,

            title=
                candidate.title,

            family=
                "group_comparison",

            dataset_id=
                dataset_id,

            dataset=
                dataset_name,

            execution_status=
                "failed",

            variables=[
                group_column,
                measure_column,
            ],

            warnings=[
                (
                    "One or more columns required "
                    "for the aggregate breakdown "
                    "are missing."
                )
            ],

            limitations=
                candidate.limitations,

            execution_rule_version=
                DERIVED_DISCOVERY_POLICY_VERSION,
        )


    composition = (
        calculate_aggregate_composition(
            dataframe,
            group_column=
                group_column,
            measure_column=
                measure_column,
        )
    )


    if composition is None:
        return SingleDatasetExecutedAnalysis(
            analysis_id=
                candidate.analysis_id,

            title=
                candidate.title,

            family=
                "group_comparison",

            dataset_id=
                dataset_id,

            dataset=
                dataset_name,

            execution_status=
                "needs_specialized_method",

            variables=[
                group_column,
                measure_column,
            ],

            valid_observations=
                int(
                    len(
                        dataframe
                    )
                ),

            warnings=[
                (
                    "The aggregate breakdown "
                    "requires one valid aggregated "
                    "row per categorical level."
                )
            ],

            limitations=
                candidate.limitations,

            execution_rule_version=
                DERIVED_DISCOVERY_POLICY_VERSION,
        )


    chart_data: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for item in (
        composition[
            "rows"
        ]
    ):
        value = float(
            item[
                "value"
            ]
        )


        chart_data.append(
            {
                "group":
                    item[
                        "group"
                    ],

                "rank":
                    item[
                        "rank"
                    ],

                "n":
                    1,

                "value":
                    value,

                "share":
                    item[
                        "share"
                    ],

                "mean":
                    value,

                "median":
                    value,

                "q1":
                    value,

                "q3":
                    value,
            }
        )


    total = float(
        composition[
            "total_value"
        ]
    )


    top_group = (
        composition[
            "top_group"
        ]
    )


    top_share = float(
        composition[
            "top_share"
        ]
    )


    return SingleDatasetExecutedAnalysis(
        analysis_id=
            candidate.analysis_id,

        title=
            candidate.title,

        family=
            "group_comparison",

        dataset_id=
            dataset_id,

        dataset=
            dataset_name,

        execution_status=
            "complete",

        variables=[
            group_column,
            measure_column,
        ],

        valid_observations=
            int(
                composition[
                    "group_count"
                ]
            ),

        summary=[
            (
                f"{composition['group_count']} "
                "catégorie(s) composent le total "
                "observé."
            ),

            (
                f"Le total de {measure_column} "
                f"est de {total:.2f}."
            ),

            (
                f"La catégorie {top_group} "
                f"représente "
                f"{top_share * 100:.2f}% "
                "du total."
            ),
        ],

        metrics={
            "analysis_semantics":
                "aggregate_breakdown",

            "group_column":
                group_column,

            "measure_column":
                measure_column,

            "group_count":
                composition[
                    "group_count"
                ],

            "total_value":
                total,

            "top_group":
                top_group,

            "top_value":
                composition[
                    "top_value"
                ],

            "top_share":
                top_share,

            "min_share":
                composition[
                    "min_share"
                ],

            "max_share":
                composition[
                    "max_share"
                ],

            "share_spread":
                composition[
                    "share_spread"
                ],

            "ranking_compatibility_family":
                "group_comparison",

            "semantic_family":
                "aggregate_breakdown",

            "derived_policy_version":
                DERIVED_DISCOVERY_POLICY_VERSION,
        },

        chart_type=
            "bar",

        chart_data=
            chart_data,

        warnings=[],

        limitations=
            candidate.limitations,

        execution_rule_version=
            DERIVED_DISCOVERY_POLICY_VERSION,
    )


# ============================================================
# EXECUTION INJECTION
# ============================================================

def inject_aggregate_breakdown_execution(
    execution: SingleDatasetExecutionReport,
    *,
    discovery: AnalysisDiscoveryReport,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> SingleDatasetExecutionReport:
    dataset_map = (
        build_dataset_map(
            datasets
        )
    )


    breakdown_candidates = [
        candidate

        for candidate
        in discovery.candidates

        if (
            candidate.scope
            ==
            "single_dataset"
            and
            candidate.family
            ==
            "aggregate_breakdown"
        )
    ]


    if not breakdown_candidates:
        return execution


    breakdown_ids = {
        candidate.analysis_id

        for candidate
        in breakdown_candidates
    }


    retained_results = [
        result

        for result
        in execution.results

        if (
            result.analysis_id
            not in breakdown_ids
        )
    ]


    custom_results: list[
        SingleDatasetExecutedAnalysis
    ] = []


    for candidate in (
        breakdown_candidates
    ):
        dataset_id = (
            candidate.dataset_ids[
                0
            ]
        )


        dataset = (
            dataset_map.get(
                dataset_id
            )
        )


        if dataset is None:
            custom_results.append(
                SingleDatasetExecutedAnalysis(
                    analysis_id=
                        candidate.analysis_id,

                    title=
                        candidate.title,

                    family=
                        "group_comparison",

                    dataset_id=
                        dataset_id,

                    dataset=
                        "unknown",

                    execution_status=
                        "failed",

                    warnings=[
                        (
                            "The derived dataset "
                            "required for the "
                            "aggregate breakdown "
                            "is unavailable."
                        )
                    ],

                    limitations=
                        candidate.limitations,

                    execution_rule_version=
                        DERIVED_DISCOVERY_POLICY_VERSION,
                )
            )

            continue


        custom_results.append(
            execute_aggregate_breakdown(
                candidate,
                dataset=
                    dataset,
            )
        )


    execution.results = [
        *retained_results,
        *custom_results,
    ]


    def count_status(
        status: str,
    ) -> int:
        return sum(
            1

            for result
            in execution.results

            if (
                result.execution_status
                ==
                status
            )
        )


    execution.candidate_count = (
        len(
            execution.results
        )
    )


    execution.complete_count = (
        count_status(
            "complete"
        )
    )


    execution.descriptive_only_count = (
        count_status(
            "descriptive_only"
        )
    )


    execution.needs_specialized_method_count = (
        count_status(
            "needs_specialized_method"
        )
    )


    execution.skipped_count = (
        count_status(
            "skipped"
        )
    )


    execution.failed_count = (
        count_status(
            "failed"
        )
    )


    execution.executor_notes.append(
        (
            "Aggregate breakdown execution was "
            "provided by "
            f"{DERIVED_DISCOVERY_POLICY_VERSION}."
        )
    )


    return execution


# ============================================================
# RANKING POLICY
# ============================================================

def apply_aggregate_breakdown_ranking_policy(
    ranking: Any,
    *,
    discovery: AnalysisDiscoveryReport,
) -> Any:
    candidates = {
        candidate.analysis_id:
            candidate

        for candidate
        in discovery.candidates

        if (
            candidate.family
            ==
            "aggregate_breakdown"
        )
    }


    if not candidates:
        return ranking


    for finding in (
        ranking.findings
    ):
        candidate = (
            candidates.get(
                finding.analysis_id
            )
        )


        if candidate is None:
            continue


        if (
            finding.execution_status
            !=
            "complete"
        ):
            continue


        share_spread_raw = (
            candidate
            .observed_signals
            .get(
                "share_spread"
            )
        )


        try:
            share_spread = float(
                share_spread_raw
            )

        except (
            TypeError,
            ValueError,
        ):
            share_spread = 0.0


        signal_score = min(
            100.0,
            max(
                45.0,
                45.0
                +
                share_spread
                *
                250.0,
            ),
        )


        execution_confidence = (
            95.0
        )


        score = (
            candidate.priority_score
            *
            0.50
            +
            signal_score
            *
            0.30
            +
            execution_confidence
            *
            0.20
        )


        finding.signal_type = (
            "group_difference"
        )


        finding.signal_score = round(
            signal_score,
            2,
        )


        finding.coverage_score = (
            100.0
        )


        finding.consistency_score = (
            50.0
        )


        finding.discovery_priority_score = (
            candidate.priority_score
        )


        finding.execution_confidence_score = (
            execution_confidence
        )


        finding.interestingness_score = round(
            min(
                100.0,
                score,
            ),
            2,
        )


        finding.direction = (
            "mixed"
        )


        finding.strength = (
            "aggregate_breakdown"
        )


        if (
            candidate.priority_score
            >=
            85.0
            and
            signal_score
            >=
            60.0
        ):
            finding.tier = (
                "key_finding"
            )

        elif (
            finding.interestingness_score
            >=
            55.0
        ):
            finding.tier = (
                "supporting_finding"
            )

        else:
            finding.tier = (
                "supplementary"
            )


        reason = (
            "La répartition du total par "
            "catégorie est directement "
            "interprétable."
        )


        if (
            reason
            not in finding.reasons
        ):
            finding.reasons.append(
                reason
            )


    tier_order = {
        "key_finding":
            3,

        "supporting_finding":
            2,

        "supplementary":
            1,

        "blocked":
            0,
    }


    ranking.findings.sort(
        key=lambda finding: (
            tier_order.get(
                finding.tier,
                0,
            ),

            finding.interestingness_score,
        ),
        reverse=True,
    )


    for index, finding in enumerate(
        ranking.findings,
        start=1,
    ):
        finding.rank = (
            index
        )


    ranking.ranked_count = (
        len(
            ranking.findings
        )
    )


    ranking.key_finding_count = sum(
        1

        for finding
        in ranking.findings

        if (
            finding.tier
            ==
            "key_finding"
        )
    )


    ranking.supporting_finding_count = sum(
        1

        for finding
        in ranking.findings

        if (
            finding.tier
            ==
            "supporting_finding"
        )
    )


    ranking.supplementary_count = sum(
        1

        for finding
        in ranking.findings

        if (
            finding.tier
            ==
            "supplementary"
        )
    )


    ranking.blocked_count = sum(
        1

        for finding
        in ranking.findings

        if (
            finding.tier
            ==
            "blocked"
        )
    )


    ranking.ranking_notes.append(
        (
            "Aggregate breakdown findings use "
            "a dedicated derived-view ranking "
            "policy while retaining temporary "
            "composer compatibility."
        )
    )


    return ranking


# ============================================================
# FINAL REPORT NORMALIZATION
# ============================================================

def normalize_report_aggregate_families(
    report: Any,
    *,
    discovery: AnalysisDiscoveryReport,
) -> Any:
    aggregate_ids = {
        candidate.analysis_id

        for candidate
        in discovery.candidates

        if (
            candidate.family
            ==
            "aggregate_breakdown"
        )
    }


    if not aggregate_ids:
        return report


    collections = [
        getattr(
            report,
            "main_findings",
            [],
        ),

        getattr(
            report,
            "additional_findings",
            [],
        ),

        getattr(
            report,
            "diagnostics",
            [],
        ),

        getattr(
            report,
            "context_analyses",
            [],
        ),

        getattr(
            report,
            "blocked_analyses",
            [],
        ),
    ]


    normalized_count = 0


    for collection in collections:
        for finding in collection:
            if (
                finding.analysis_id
                in aggregate_ids
            ):
                finding.family = (
                    "aggregate_breakdown"
                )

                normalized_count += 1


    methodology_notes = getattr(
        report,
        "methodology_notes",
        None,
    )


    if isinstance(
        methodology_notes,
        list,
    ):
        methodology_notes.append(
            (
                "Les répartitions agrégées sont "
                "calculées directement sur des "
                "vues dont le grain a été "
                "matérialisé et audité ; elles "
                "ne sont pas interprétées comme "
                "des tests inférentiels."
            )
        )


        methodology_notes.append(
            (
                f"{normalized_count} résultat(s) "
                "de répartition agrégée ont été "
                "normalisés dans le rapport final."
            )
        )


    return report