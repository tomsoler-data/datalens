from __future__ import annotations

import itertools
import re
import unicodedata

from app.ingestion.schemas import (
    DatasetColumnManifest,
    DatasetManifest,
    MultiDatasetIngestion,
)

from app.planning.schemas import (
    AdditionalDataSuggestion,
    AnalysisCandidate,
    AnalysisPlanReport,
    CrossDatasetOpportunity,
    PlannedVariable,
)


MAX_ANALYSES_PER_DATASET = 8

MAX_RECOMMENDED_ANALYSES = 16


CONCEPT_TERMS = {
    "population": {
        "population",
        "inhabitants",
        "habitants",
    },

    "water": {
        "water",
        "drinking",
        "wash",
        "eau",
        "potable",
        "access",
        "acces",
    },

    "health": {
        "mortality",
        "death",
        "deaths",
        "health",
        "mortalite",
        "deces",
        "sante",
        "disease",
        "maladie",
    },

    "political": {
        "political",
        "stability",
        "politique",
        "stabilite",
    },

    "economic": {
        "gdp",
        "income",
        "revenue",
        "wealth",
        "economic",
        "economique",
        "pib",
        "revenu",
    },

    "urbanization": {
        "urban",
        "rural",
        "urbanization",
        "urbanisation",
    },

    "investment": {
        "investment",
        "cost",
        "budget",
        "project",
        "infrastructure",
        "investissement",
        "cout",
        "projet",
    },
}


GEOGRAPHIC_TERMS = {
    "country",
    "countries",
    "pays",
    "region",
    "regions",
    "continent",
    "state",
    "province",
}


def normalize_text(
    value: str,
) -> str:
    normalized = (
        unicodedata
        .normalize(
            "NFKD",
            value,
        )
        .encode(
            "ascii",
            "ignore",
        )
        .decode(
            "ascii",
        )
        .lower()
    )

    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        normalized,
    )

    return (
        " ".join(
            normalized.split()
        )
    )


def tokenize(
    value: str,
) -> set[
    str
]:
    return set(
        normalize_text(
            value
        ).split()
    )


def combined_dataset_tokens(
    manifest: DatasetManifest,
) -> set[
    str
]:
    tokens = tokenize(
        manifest.filename
    )

    for column in (
        manifest.columns
    ):
        tokens.update(
            tokenize(
                column.name
            )
        )

    return tokens


def detect_concepts(
    manifests: list[
        DatasetManifest
    ],
) -> set[
    str
]:
    all_tokens: set[
        str
    ] = set()

    for manifest in (
        manifests
    ):
        all_tokens.update(
            combined_dataset_tokens(
                manifest
            )
        )

    detected: set[
        str
    ] = set()

    for (
        concept,
        terms,
    ) in CONCEPT_TERMS.items():
        if (
            all_tokens
            &
            terms
        ):
            detected.add(
                concept
            )

    return detected


def objective_bonus(
    objective: str | None,
    columns: list[
        str
    ],
) -> int:
    if not objective:
        return 0

    objective_tokens = (
        tokenize(
            objective
        )
    )

    if not objective_tokens:
        return 0

    column_tokens: set[
        str
    ] = set()

    for column in (
        columns
    ):
        column_tokens.update(
            tokenize(
                column
            )
        )

    direct_matches = (
        objective_tokens
        &
        column_tokens
    )

    bonus = min(
        len(
            direct_matches
        )
        * 7,
        21,
    )

    for terms in (
        CONCEPT_TERMS
        .values()
    ):
        if (
            objective_tokens
            &
            terms
            and
            column_tokens
            &
            terms
        ):
            bonus += 6

    return min(
        bonus,
        30,
    )


def get_columns_by_kind(
    manifest: DatasetManifest,
    kind: str,
) -> list[
    DatasetColumnManifest
]:
    return [
        column
        for column
        in manifest.columns
        if (
            column.analysis_kind
            ==
            kind
        )
    ]


def find_geographic_column(
    manifest: DatasetManifest,
) -> (
    DatasetColumnManifest
    | None
):
    for column in (
        manifest.columns
    ):
        tokens = tokenize(
            column.name
        )

        if (
            tokens
            &
            GEOGRAPHIC_TERMS
        ):
            return column

    return None


def build_time_series_candidates(
    manifest: DatasetManifest,
    objective: str | None,
) -> list[
    AnalysisCandidate
]:
    temporal_columns = (
        get_columns_by_kind(
            manifest,
            "temporal",
        )
    )

    quantitative_columns = (
        get_columns_by_kind(
            manifest,
            "quantitative",
        )
    )

    if (
        not temporal_columns
        or
        not quantitative_columns
    ):
        return []

    time_column = (
        temporal_columns[
            0
        ]
    )

    geographic_column = (
        find_geographic_column(
            manifest
        )
    )

    candidates: list[
        AnalysisCandidate
    ] = []

    for (
        index,
        value_column,
    ) in enumerate(
        quantitative_columns[
            :4
        ],
        start=1,
    ):
        variables = [
            PlannedVariable(
                column=
                    time_column.name,
                role=
                    "time",
                analysis_kind=
                    time_column.analysis_kind,
            ),

            PlannedVariable(
                column=
                    value_column.name,
                role=
                    "value",
                analysis_kind=
                    value_column.analysis_kind,
            ),
        ]

        if geographic_column:
            variables.append(
                PlannedVariable(
                    column=
                        geographic_column.name,
                    role=
                        "group",
                    analysis_kind=
                        geographic_column.analysis_kind,
                )
            )

        score = (
            88
            +
            objective_bonus(
                objective,
                [
                    time_column.name,
                    value_column.name,
                ],
            )
        )

        reasons = [
            (
                f"{time_column.name} was "
                "identified as temporal."
            ),
            (
                f"{value_column.name} was "
                "identified as quantitative."
            ),
            (
                "A line chart is appropriate "
                "for examining change across "
                "ordered time values."
            ),
        ]

        if geographic_column:
            reasons.append(
                (
                    f"{geographic_column.name} "
                    "can be used as a grouping "
                    "or filtering dimension."
                )
            )

        candidates.append(
            AnalysisCandidate(
                analysis_id=(
                    f"{manifest.dataset_id}:"
                    f"time:{index:02d}"
                ),

                dataset_id=
                    manifest.dataset_id,

                dataset_filename=
                    manifest.filename,

                title=(
                    "Évolution de "
                    f"{value_column.name}"
                ),

                family=
                    "time_series",

                priority_score=
                    min(
                        score,
                        100,
                    ),

                readiness=
                    "planned",

                variables=
                    variables,

                chart_type=
                    "line",

                statistical_strategy=
                    "descriptive_time_series",

                reasons=
                    reasons,

                limitations=[
                    (
                        "Trend modelling and "
                        "forecasting are not yet "
                        "executed automatically."
                    )
                ],
            )
        )

    return candidates


def association_quality_score(
    left: DatasetColumnManifest,
    right: DatasetColumnManifest,
) -> int:
    missing_penalty = int(
        (
            left.missing_ratio
            +
            right.missing_ratio
        )
        *
        20
    )

    score = (
        82
        -
        missing_penalty
    )

    if (
        left.unique_candidate
        or
        right.unique_candidate
    ):
        score -= 8

    return max(
        score,
        40,
    )


def build_quantitative_association_candidates(
    manifest: DatasetManifest,
    objective: str | None,
) -> list[
    AnalysisCandidate
]:
    quantitative_columns = (
        get_columns_by_kind(
            manifest,
            "quantitative",
        )
    )

    if (
        len(
            quantitative_columns
        )
        <
        2
    ):
        return []

    pairs = list(
        itertools.combinations(
            quantitative_columns,
            2,
        )
    )

    candidates: list[
        AnalysisCandidate
    ] = []

    for (
        index,
        (
            left,
            right,
        ),
    ) in enumerate(
        pairs[
            :8
        ],
        start=1,
    ):
        score = (
            association_quality_score(
                left,
                right,
            )
            +
            objective_bonus(
                objective,
                [
                    left.name,
                    right.name,
                ],
            )
        )

        chart_type = (
            "hexbin"
            if (
                manifest.row_count
                >
                3000
            )
            else
            "scatter"
        )

        candidates.append(
            AnalysisCandidate(
                analysis_id=(
                    f"{manifest.dataset_id}:"
                    f"association:{index:02d}"
                ),

                dataset_id=
                    manifest.dataset_id,

                dataset_filename=
                    manifest.filename,

                title=(
                    "Relation entre "
                    f"{left.name} et "
                    f"{right.name}"
                ),

                family=
                    "quantitative_association",

                priority_score=
                    min(
                        score,
                        100,
                    ),

                readiness=
                    "executable_now",

                variables=[
                    PlannedVariable(
                        column=
                            left.name,
                        role=
                            "x",
                        analysis_kind=
                            left.analysis_kind,
                    ),

                    PlannedVariable(
                        column=
                            right.name,
                        role=
                            "y",
                        analysis_kind=
                            right.analysis_kind,
                    ),
                ],

                chart_type=
                    chart_type,

                statistical_strategy=(
                    "automatic_correlation_"
                    "decision_engine"
                ),

                reasons=[
                    (
                        "Both variables are "
                        "quantitative."
                    ),
                    (
                        "The current statistical "
                        "decision engine can "
                        "evaluate this association "
                        "automatically."
                    ),
                    (
                        "Hexbin is preferred for "
                        "dense datasets."
                        if (
                            chart_type
                            ==
                            "hexbin"
                        )
                        else
                        "A scatter plot preserves "
                        "the paired observations."
                    ),
                ],

                limitations=[
                    (
                        "Statistical method selection "
                        "still depends on the "
                        "correlation decision engine."
                    )
                ],
            )
        )

    return candidates


def build_group_comparison_candidates(
    manifest: DatasetManifest,
    objective: str | None,
) -> list[
    AnalysisCandidate
]:
    categorical_columns = [
        column
        for column
        in get_columns_by_kind(
            manifest,
            "categorical",
        )
        if (
            2
            <=
            column.unique_count
            <=
            20
        )
    ]

    quantitative_columns = (
        get_columns_by_kind(
            manifest,
            "quantitative",
        )
    )

    candidates: list[
        AnalysisCandidate
    ] = []

    index = 0

    for category in (
        categorical_columns[
            :3
        ]
    ):
        for value in (
            quantitative_columns[
                :3
            ]
        ):
            index += 1

            score = (
                74
                +
                objective_bonus(
                    objective,
                    [
                        category.name,
                        value.name,
                    ],
                )
            )

            candidates.append(
                AnalysisCandidate(
                    analysis_id=(
                        f"{manifest.dataset_id}:"
                        f"groups:{index:02d}"
                    ),

                    dataset_id=
                        manifest.dataset_id,

                    dataset_filename=
                        manifest.filename,

                    title=(
                        f"{value.name} selon "
                        f"{category.name}"
                    ),

                    family=
                        "group_comparison",

                    priority_score=
                        min(
                            score,
                            100,
                        ),

                    readiness=
                        "planned",

                    variables=[
                        PlannedVariable(
                            column=
                                category.name,
                            role=
                                "group",
                            analysis_kind=
                                category.analysis_kind,
                        ),

                        PlannedVariable(
                            column=
                                value.name,
                            role=
                                "value",
                            analysis_kind=
                                value.analysis_kind,
                        ),
                    ],

                    chart_type=
                        "boxplot",

                    statistical_strategy=(
                        "automatic_group_"
                        "comparison_engine"
                    ),

                    reasons=[
                        (
                            f"{category.name} has "
                            f"{category.unique_count} "
                            "groups."
                        ),
                        (
                            f"{value.name} is "
                            "quantitative."
                        ),
                        (
                            "A boxplot can compare "
                            "the distribution across "
                            "groups without reducing "
                            "the data to a single "
                            "summary statistic."
                        ),
                    ],

                    limitations=[
                        (
                            "The automatic group "
                            "comparison statistical "
                            "executor is not yet "
                            "implemented."
                        )
                    ],
                )
            )

    return candidates


def build_categorical_association_candidates(
    manifest: DatasetManifest,
    objective: str | None,
) -> list[
    AnalysisCandidate
]:
    categorical_columns = [
        column
        for column
        in get_columns_by_kind(
            manifest,
            "categorical",
        )
        if (
            2
            <=
            column.unique_count
            <=
            20
        )
    ]

    if (
        len(
            categorical_columns
        )
        <
        2
    ):
        return []

    candidates: list[
        AnalysisCandidate
    ] = []

    pairs = itertools.combinations(
        categorical_columns[
            :5
        ],
        2,
    )

    for (
        index,
        (
            left,
            right,
        ),
    ) in enumerate(
        pairs,
        start=1,
    ):
        score = (
            66
            +
            objective_bonus(
                objective,
                [
                    left.name,
                    right.name,
                ],
            )
        )

        candidates.append(
            AnalysisCandidate(
                analysis_id=(
                    f"{manifest.dataset_id}:"
                    f"categorical:{index:02d}"
                ),

                dataset_id=
                    manifest.dataset_id,

                dataset_filename=
                    manifest.filename,

                title=(
                    "Association entre "
                    f"{left.name} et "
                    f"{right.name}"
                ),

                family=
                    "categorical_association",

                priority_score=
                    min(
                        score,
                        100,
                    ),

                readiness=
                    "planned",

                variables=[
                    PlannedVariable(
                        column=
                            left.name,
                        role=
                            "category",
                        analysis_kind=
                            left.analysis_kind,
                    ),

                    PlannedVariable(
                        column=
                            right.name,
                        role=
                            "category",
                        analysis_kind=
                            right.analysis_kind,
                    ),
                ],

                chart_type=
                    "heatmap",

                statistical_strategy=(
                    "chi_square_or_fisher_"
                    "decision_engine"
                ),

                reasons=[
                    (
                        "Both variables are "
                        "categorical."
                    ),
                    (
                        "Their cardinalities are "
                        "low enough for a "
                        "contingency analysis."
                    ),
                ],

                limitations=[
                    (
                        "Expected cell counts must "
                        "be evaluated before "
                        "selecting chi-square or "
                        "Fisher inference."
                    )
                ],
            )
        )

    return candidates


def build_distribution_candidates(
    manifest: DatasetManifest,
    objective: str | None,
) -> list[
    AnalysisCandidate
]:
    quantitative_columns = (
        get_columns_by_kind(
            manifest,
            "quantitative",
        )
    )

    candidates: list[
        AnalysisCandidate
    ] = []

    for (
        index,
        column,
    ) in enumerate(
        quantitative_columns[
            :3
        ],
        start=1,
    ):
        score = (
            48
            +
            objective_bonus(
                objective,
                [
                    column.name
                ],
            )
        )

        candidates.append(
            AnalysisCandidate(
                analysis_id=(
                    f"{manifest.dataset_id}:"
                    f"distribution:{index:02d}"
                ),

                dataset_id=
                    manifest.dataset_id,

                dataset_filename=
                    manifest.filename,

                title=(
                    "Distribution de "
                    f"{column.name}"
                ),

                family=
                    "distribution",

                priority_score=
                    min(
                        score,
                        100,
                    ),

                readiness=
                    "planned",

                variables=[
                    PlannedVariable(
                        column=
                            column.name,
                        role=
                            "value",
                        analysis_kind=
                            column.analysis_kind,
                    ),
                ],

                chart_type=
                    "histogram",

                statistical_strategy=
                    "descriptive_distribution",

                reasons=[
                    (
                        f"{column.name} is "
                        "quantitative."
                    ),
                    (
                        "Distribution inspection "
                        "supports later statistical "
                        "interpretation."
                    ),
                ],
            )
        )

    return candidates


def build_dataset_candidates(
    manifest: DatasetManifest,
    objective: str | None,
) -> list[
    AnalysisCandidate
]:
    candidates = (
        build_time_series_candidates(
            manifest,
            objective,
        )
        +
        build_quantitative_association_candidates(
            manifest,
            objective,
        )
        +
        build_group_comparison_candidates(
            manifest,
            objective,
        )
        +
        build_categorical_association_candidates(
            manifest,
            objective,
        )
        +
        build_distribution_candidates(
            manifest,
            objective,
        )
    )

    candidates.sort(
        key=lambda candidate:
            candidate.priority_score,
        reverse=True,
    )

    return candidates[
        :MAX_ANALYSES_PER_DATASET
    ]


def build_cross_dataset_opportunities(
    manifests: list[
        DatasetManifest
    ],
) -> list[
    CrossDatasetOpportunity
]:
    opportunities: list[
        CrossDatasetOpportunity
    ] = []

    opportunity_index = 0

    for (
        left,
        right,
    ) in itertools.combinations(
        manifests,
        2,
    ):
        left_columns = {
            normalize_text(
                column.name
            ):
            column.name
            for column
            in left.columns
        }

        right_columns = {
            normalize_text(
                column.name
            ):
            column.name
            for column
            in right.columns
        }

        shared_normalized = (
            set(
                left_columns
            )
            &
            set(
                right_columns
            )
        )

        if not shared_normalized:
            continue

        shared_columns = [
            left_columns[
                normalized
            ]
            for normalized
            in sorted(
                shared_normalized
            )
        ]

        opportunity_index += 1

        opportunities.append(
            CrossDatasetOpportunity(
                opportunity_id=(
                    "relationship:"
                    f"{opportunity_index:04d}"
                ),

                dataset_ids=[
                    left.dataset_id,
                    right.dataset_id,
                ],

                dataset_filenames=[
                    left.filename,
                    right.filename,
                ],

                shared_columns=
                    shared_columns,

                reason=(
                    "The datasets share one or "
                    "more column names that may "
                    "support a relationship or "
                    "join. Cardinality and value "
                    "coverage must be validated "
                    "before combining them."
                ),

                requires_relationship_validation=
                    True,
            )
        )

    return opportunities


def build_additional_data_suggestions(
    manifests: list[
        DatasetManifest
    ],
) -> list[
    AdditionalDataSuggestion
]:
    concepts = (
        detect_concepts(
            manifests
        )
    )

    suggestions: list[
        AdditionalDataSuggestion
    ] = []

    suggestion_index = 0

    def add_suggestion(
        title: str,
        priority: str,
        rationale: str,
        fields: list[
            str
        ],
    ) -> None:
        nonlocal suggestion_index

        suggestion_index += 1

        suggestions.append(
            AdditionalDataSuggestion(
                suggestion_id=(
                    "data_suggestion:"
                    f"{suggestion_index:04d}"
                ),

                title=
                    title,

                priority=
                    priority,

                rationale=
                    rationale,

                example_fields=
                    fields,

                required_for_current_analysis=
                    False,
            )
        )

    if (
        "water"
        in concepts
        and
        "economic"
        not in concepts
    ):
        add_suggestion(
            title=(
                "Ajouter des indicateurs "
                "économiques"
            ),

            priority=
                "medium",

            rationale=(
                "Des indicateurs économiques "
                "permettraient d’examiner si "
                "les différences d’accès aux "
                "services sont associées au "
                "niveau de développement."
            ),

            fields=[
                "PIB par habitant",
                "revenu médian",
                "taux de pauvreté",
            ],
        )

    if (
        "water"
        in concepts
        and
        "health"
        not in concepts
    ):
        add_suggestion(
            title=(
                "Ajouter des indicateurs "
                "sanitaires"
            ),

            priority=
                "high",

            rationale=(
                "Des données sanitaires "
                "permettraient d’étudier les "
                "conséquences potentielles "
                "d’un accès insuffisant à "
                "l’eau potable."
            ),

            fields=[
                "mortalité liée à l’eau",
                "maladies hydriques",
                "hospitalisations",
            ],
        )

    if (
        "population"
        in concepts
        and
        "urbanization"
        not in concepts
    ):
        add_suggestion(
            title=(
                "Ajouter le niveau "
                "d’urbanisation"
            ),

            priority=
                "medium",

            rationale=(
                "La distinction entre zones "
                "urbaines et rurales peut "
                "expliquer une partie des "
                "différences de couverture "
                "des infrastructures."
            ),

            fields=[
                "part de population urbaine",
                "part de population rurale",
                "densité de population",
            ],
        )

    if (
        "water"
        in concepts
        and
        "investment"
        not in concepts
    ):
        add_suggestion(
            title=(
                "Ajouter les investissements "
                "et infrastructures"
            ),

            priority=
                "medium",

            rationale=(
                "Ces données permettraient de "
                "passer d’une analyse du besoin "
                "à une analyse de faisabilité "
                "et de priorisation opérationnelle."
            ),

            fields=[
                "budget infrastructure",
                "coût des projets",
                "projets existants",
                "capacité de traitement",
            ],
        )

    return suggestions


def build_analysis_plan(
    ingestion: MultiDatasetIngestion,
    *,
    objective: str | None = None,
) -> AnalysisPlanReport:
    all_candidates: list[
        AnalysisCandidate
    ] = []

    for manifest in (
        ingestion.datasets
    ):
        all_candidates.extend(
            build_dataset_candidates(
                manifest,
                objective,
            )
        )

    all_candidates.sort(
        key=lambda candidate:
            candidate.priority_score,
        reverse=True,
    )

    selected_candidates = (
        all_candidates[
            :MAX_RECOMMENDED_ANALYSES
        ]
    )

    executable_count = sum(
        1
        for candidate
        in selected_candidates
        if (
            candidate.readiness
            ==
            "executable_now"
        )
    )

    planner_notes = [
        (
            f"{len(selected_candidates)} "
            "analysis candidate(s) were "
            "selected from the highest-priority "
            "deterministic rules."
        ),
        (
            f"{executable_count} candidate(s) "
            "can currently be executed by the "
            "implemented statistical pipeline."
        ),
        (
            "Candidate ranking is not evidence "
            "of causality or business importance."
        ),
    ]

    if objective:
        planner_notes.append(
            (
                "The user objective was used "
                "only to increase the priority "
                "of analyses whose variables "
                "or concepts overlap with the "
                "request."
            )
        )
    else:
        planner_notes.append(
            (
                "No objective was supplied, so "
                "the planner used automatic "
                "exploratory prioritization."
            )
        )

    return AnalysisPlanReport(
        objective=(
            objective.strip()
            if (
                objective
                and
                objective.strip()
            )
            else
            None
        ),

        dataset_count=
            ingestion.dataset_count,

        total_rows=
            ingestion.total_rows,

        recommended_analyses=
            selected_candidates,

        cross_dataset_opportunities=(
            build_cross_dataset_opportunities(
                ingestion.datasets
            )
        ),

        additional_data_suggestions=(
            build_additional_data_suggestions(
                ingestion.datasets
            )
        ),

        planner_notes=
            planner_notes,
    )