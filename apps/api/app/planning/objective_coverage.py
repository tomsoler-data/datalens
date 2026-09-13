from __future__ import annotations


from dataclasses import (
    dataclass,
)

import re
import unicodedata

from typing import (
    Any,
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.planning.analytical_contract import (
    AnalyticalContract,
)


# ============================================================
# VERSION
# ============================================================

OBJECTIVE_COVERAGE_RULE_VERSION = (
    "objective_coverage_v0.3"
)


# ============================================================
# TYPES
# ============================================================

ObjectiveCoverageStatus = Literal[
    "complete",
    "incomplete",
    "not_applicable",
]


ObjectiveRequirementType = Literal[
    "metric",
    "dimension",
    "column",
    "derived_metric",
]


# ============================================================
# ANALYTICAL INTENT COVERAGE
# DATALENS_OBJECTIVE_INTENT_COVERAGE_V0_1
# ============================================================

OBJECTIVE_INTENT_COVERAGE_RULE_VERSION = (
    "objective_intent_coverage_v0.1"
)


ObjectiveIntentKind = Literal[
    "scalar_total",
    "monthly_time_series",
    "categorical_breakdown",
    "entity_ranking",
]


# ============================================================
# REPORT
# ============================================================

class ObjectiveCoverageRequirement(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    requirement_id: str

    concept: str

    requirement_type: (
        ObjectiveRequirementType
    )

    requested_phrases: list[
        str
    ] = Field(
        default_factory=list
    )

    candidate_columns: list[
        str
    ] = Field(
        default_factory=list
    )

    allowed_roles: list[
        str
    ] = Field(
        default_factory=list
    )

    required_aggregation: (
        str
        | None
    ) = None

    covered: bool

    covered_by_contract_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    notes: list[
        str
    ] = Field(
        default_factory=list
    )



# ============================================================
# METRIC x DIMENSION TOPOLOGY
# DATALENS_OBJECTIVE_COVERAGE_TOPOLOGY_V0_2
# ============================================================

class ObjectiveCoverageTopologyRequirement(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    topology_id: str

    metric_concept: str

    required_dimension_concepts: list[
        str
    ] = Field(
        default_factory=list
    )

    covered: bool

    covered_by_contract_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    notes: list[
        str
    ] = Field(
        default_factory=list
    )


class ObjectiveCoverageIntentRequirement(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    intent_id: str

    intent_kind: ObjectiveIntentKind

    metric_concept: str

    candidate_metric_columns: list[
        str
    ] = Field(
        default_factory=list
    )

    requested_phrases: list[
        str
    ] = Field(
        default_factory=list
    )

    required_family: (
        str
        | None
    ) = None

    required_grain: (
        str
        | None
    ) = None

    required_dimension_concepts: list[
        str
    ] = Field(
        default_factory=list
    )

    required_aggregation: (
        str
        | None
    ) = None

    ranking_order: (
        str
        | None
    ) = None

    ranking_limit: (
        int
        | None
    ) = None

    covered: bool = False

    covered_by_contract_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    notes: list[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        OBJECTIVE_INTENT_COVERAGE_RULE_VERSION
    )


class ObjectiveCoverageReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    status: ObjectiveCoverageStatus

    requirement_count: int

    covered_count: int

    missing_count: int

    requirements: list[
        ObjectiveCoverageRequirement
    ] = Field(
        default_factory=list
    )

    topology_requirement_count: int = 0

    topology_covered_count: int = 0

    topology_missing_count: int = 0

    topology_requirements: list[
        ObjectiveCoverageTopologyRequirement
    ] = Field(
        default_factory=list
    )

    intent_requirement_count: int = 0

    intent_covered_count: int = 0

    intent_missing_count: int = 0

    intent_requirements: list[
        ObjectiveCoverageIntentRequirement
    ] = Field(
        default_factory=list
    )

    notes: list[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        OBJECTIVE_COVERAGE_RULE_VERSION
    )


# ============================================================
# INTERNAL SEMANTIC SPEC
# ============================================================

@dataclass(
    frozen=True
)
class SemanticRequirementSpec:
    requirement_id: str

    concept: str

    requirement_type: (
        ObjectiveRequirementType
    )

    phrases: tuple[
        str,
        ...
    ]

    candidate_column_names: tuple[
        str,
        ...
    ]

    allowed_roles: tuple[
        str,
        ...
    ] = ()

    required_aggregation: (
        str
        | None
    ) = None


# ============================================================
# CONSERVATIVE SEMANTIC VOCABULARY
# ============================================================
#
# This is intentionally small.
#
# Objective Coverage is a deterministic guard, not another
# semantic planner.
#
# A concept is enforced only when the user request contains one
# of these explicit phrases.
# ============================================================

SEMANTIC_REQUIREMENT_SPECS = (
    SemanticRequirementSpec(
        requirement_id=
            "metric:revenue_total",

        concept=
            "revenue_total",

        requirement_type=
            "metric",

        phrases=(
            "chiffre d affaires",
            "revenue",
            "turnover",
        ),

        candidate_column_names=(
            "revenue",
            "turnover",
            "sales",
            "amount",
        ),

        allowed_roles=(
            "value",
            "x",
            "y",
        ),

        required_aggregation=
            "sum",
    ),

    SemanticRequirementSpec(
        requirement_id=
            "metric:return_rate",

        concept=
            "return_rate",

        requirement_type=
            "metric",

        phrases=(
            "taux de retour",
            "return rate",
        ),

        candidate_column_names=(
            "returned_order",
            "is_returned",
            "returned",
            "return_flag",
        ),

        allowed_roles=(
            "value",
            "x",
            "y",
        ),

        # Mean(Boolean) is the deterministic return rate.
        required_aggregation=
            "mean",
    ),

    # ========================================================
    # AVERAGE BASKET
    # DATALENS_OBJECTIVE_COVERAGE_AVERAGE_BASKET_V0_1
    # ========================================================

    SemanticRequirementSpec(
        requirement_id=
            "metric:average_basket",

        concept=
            "average_basket",

        requirement_type=
            "metric",

        phrases=(
            "panier moyen",
            "average basket",
        ),

        candidate_column_names=(
            "basket_amount",
        ),

        allowed_roles=(
            "value",
        ),

        required_aggregation=
            "mean",
    ),

    SemanticRequirementSpec(
        requirement_id=
            "dimension:region",

        concept=
            "region",

        requirement_type=
            "dimension",

        phrases=(
            "region",
            "regions",
        ),

        candidate_column_names=(
            "region",
        ),

        allowed_roles=(
            "group",
            "dimension",
            "x",
            "y",
        ),
    ),

    SemanticRequirementSpec(
        requirement_id=
            "dimension:channel",

        concept=
            "channel",

        requirement_type=
            "dimension",

        phrases=(
            "canal",
            "canaux",
            "channel",
            "channels",
        ),

        candidate_column_names=(
            "channel",
            "sales_channel",
            "canal",
        ),

        allowed_roles=(
            "group",
            "dimension",
            "x",
            "y",
        ),
    ),
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(
    value: str,
) -> str:
    normalized = (
        unicodedata.normalize(
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
        .casefold()
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


def contains_phrase(
    text: str,
    phrase: str,
) -> bool:
    normalized_text = (
        " "
        +
        normalize_text(
            text
        )
        +
        " "
    )

    normalized_phrase = (
        normalize_text(
            phrase
        )
    )

    if not normalized_phrase:
        return False

    return (
        (
            " "
            +
            normalized_phrase
            +
            " "
        )
        in
        normalized_text
    )


# ============================================================
# EXPLICIT DERIVED OUTPUT REQUESTS
# ============================================================


def explicit_share_of_total_phrases(
    objective: str,
) -> list[
    str
]:
    """
    Return explicit textual evidence that the user requested
    a part / percentage / share / proportion of a total.

    This detector is deliberately conservative:

    - an isolated "part" is insufficient;
    - a total request alone is insufficient;
    - ranking language alone is insufficient;
    - benchmark language alone is insufficient.

    The share term must occur before an explicit `total` token
    within a short local phrase.
    """

    normalized = (
        normalize_text(
            objective
        )
    )


    share_terms = (
        "part",
        "pourcentage",
        "share",
        "percentage",
        "proportion",
    )


    evidence: list[
        str
    ] = []


    for term in (
        share_terms
    ):

        match = re.search(
            (
                rf"\b{re.escape(term)}\b"
                rf"(?:\s+[a-z0-9]+){{0,12}}"
                rf"\s+\btotal\b"
            ),
            normalized,
        )


        if (
            match
            is None
        ):
            continue


        phrase = (
            match
            .group(
                0
            )
            .strip()
        )


        if (
            phrase

            and

            phrase
            not in
            evidence
        ):

            evidence.append(
                phrase
            )


    return evidence


def explicit_share_of_total_request(
    objective: str,
) -> bool:

    return bool(
        explicit_share_of_total_phrases(
            objective
        )
    )


# ============================================================
# CATALOG
# ============================================================

def catalog_column_names(
    catalog: Any,
) -> list[
    str
]:
    names: list[
        str
    ] = []

    seen: set[
        str
    ] = set()

    for dataset in (
        getattr(
            catalog,
            "datasets",
            [],
        )
        or []
    ):
        for column in (
            getattr(
                dataset,
                "columns",
                [],
            )
            or []
        ):
            name = str(
                getattr(
                    column,
                    "name",
                    "",
                )
                or
                ""
            ).strip()

            if not name:
                continue

            if name in seen:
                continue

            seen.add(
                name
            )

            names.append(
                name
            )

    return names


def catalog_semantic_alias_targets(
    catalog: Any,
) -> dict[
    str,
    list[
        str
    ],
]:
    """
    Resolve server-owned analytical measure aliases to the
    physical target columns they describe.

    Important safety rule:

    aliases never become executable columns themselves.

    A semantic alias is accepted only when:

    - a dataset declares target_measure_column;
    - that target exists physically in the same dataset;
    - the alias is explicitly present in the server-owned
      PlannerDatasetProfile.measure_semantic_aliases field.

    This preserves the narrow contract-local semantic bridge
    established by the planner catalog instead of rebuilding a
    broad global synonym table inside Objective Coverage.
    """

    result: dict[
        str,
        list[
            str
        ],
    ] = {}


    for dataset in (
        getattr(
            catalog,
            "datasets",
            [],
        )
        or []
    ):
        target_measure = str(
            getattr(
                dataset,
                "target_measure_column",
                "",
            )
            or
            ""
        ).strip()


        if not target_measure:
            continue


        physical_columns = {
            str(
                getattr(
                    column,
                    "name",
                    "",
                )
                or
                ""
            ).strip()

            for column
            in (
                getattr(
                    dataset,
                    "columns",
                    [],
                )
                or []
            )
        }


        if (
            target_measure
            not in
            physical_columns
        ):
            continue


        aliases = (
            getattr(
                dataset,
                "measure_semantic_aliases",
                [],
            )
            or []
        )


        for raw_alias in aliases:
            normalized_alias = (
                normalize_text(
                    str(
                        raw_alias
                        or
                        ""
                    )
                )
            )


            if not normalized_alias:
                continue


            targets = (
                result.setdefault(
                    normalized_alias,
                    [],
                )
            )


            if (
                target_measure
                not in
                targets
            ):
                targets.append(
                    target_measure
                )


    return result


def resolve_candidate_columns(
    *,
    catalog_columns: list[
        str
    ],

    candidate_names: tuple[
        str,
        ...
    ],

    semantic_alias_targets: (
        dict[
            str,
            list[
                str
            ],
        ]
        | None
    ) = None,
) -> list[
    str
]:
    normalized_catalog = {
        normalize_text(
            name
        ):
            name

        for name
        in catalog_columns
    }


    alias_targets = (
        semantic_alias_targets
        or {}
    )


    resolved: list[
        str
    ] = []


    for candidate_name in (
        candidate_names
    ):
        normalized_candidate = (
            normalize_text(
                candidate_name
            )
        )


        actual = (
            normalized_catalog.get(
                normalized_candidate
            )
        )


        if actual is not None:
            if (
                actual
                not in
                resolved
            ):
                resolved.append(
                    actual
                )


            continue


        for target_column in (
            alias_targets.get(
                normalized_candidate,
                [],
            )
        ):
            if (
                target_column
                in
                resolved
            ):
                continue


            resolved.append(
                target_column
            )


    return resolved



# ============================================================
# REQUIREMENT EXTRACTION
# ============================================================

def extract_objective_requirements(
    *,
    objective: str,
    catalog: Any,
) -> list[
    ObjectiveCoverageRequirement
]:
    normalized_objective = (
        normalize_text(
            objective
        )
    )

    catalog_columns = (
        catalog_column_names(
            catalog
        )
    )


    semantic_alias_targets = (
        catalog_semantic_alias_targets(
            catalog
        )
    )


    requirements: list[
        ObjectiveCoverageRequirement
    ] = []

    semantic_column_names: set[
        str
    ] = set()


    # --------------------------------------------------------
    # SEMANTIC BUSINESS CONCEPTS
    # --------------------------------------------------------

    for spec in (
        SEMANTIC_REQUIREMENT_SPECS
    ):
        matched_phrases = [
            phrase

            for phrase
            in spec.phrases

            if contains_phrase(
                normalized_objective,
                phrase,
            )
        ]

        if not matched_phrases:
            continue

        candidates = (
            resolve_candidate_columns(
                catalog_columns=
                    catalog_columns,

                candidate_names=
                    spec
                    .candidate_column_names,

                semantic_alias_targets=
                    semantic_alias_targets,
            )
        )

        semantic_column_names.update(
            normalize_text(
                name
            )

            for name
            in candidates
        )

        notes: list[
            str
        ] = []

        if not candidates:
            notes.append(
                (
                    "The requested concept was detected, "
                    "but no compatible physical column or "
                    "trusted server-owned analytical alias "
                    "was resolved from the current catalog."
                )
            )

        requirements.append(
            ObjectiveCoverageRequirement(
                requirement_id=
                    spec.requirement_id,

                concept=
                    spec.concept,

                requirement_type=
                    spec.requirement_type,

                requested_phrases=
                    matched_phrases,

                candidate_columns=
                    candidates,

                allowed_roles=
                    list(
                        spec.allowed_roles
                    ),

                required_aggregation=
                    spec.required_aggregation,

                covered=False,

                covered_by_contract_ids=[],

                notes=
                    notes,
            )
        )


    # --------------------------------------------------------
    # EXPLICIT SHARE OF TOTAL
    # DATALENS_OBJECTIVE_COVERAGE_SHARE_OF_TOTAL_V0_1
    # --------------------------------------------------------
    #
    # `share_of_total` is an output requirement, not a physical
    # source-column requirement.
    #
    # It therefore has:
    #
    # - no candidate physical columns;
    # - no binding role;
    # - no standalone aggregation function.
    #
    # Until a validated contract explicitly carries executable
    # share-of-total semantics, contract coverage remains false
    # and Objective Coverage must report the request incomplete.
    # --------------------------------------------------------

    share_phrases = (
        explicit_share_of_total_phrases(
            objective
        )
    )


    if (
        share_phrases
    ):

        requirements.append(
            ObjectiveCoverageRequirement(
                requirement_id=
                    "derived:share_of_total",

                concept=
                    "share_of_total",

                requirement_type=
                    "derived_metric",

                requested_phrases=
                    share_phrases,

                candidate_columns=[],

                allowed_roles=[],

                required_aggregation=None,

                covered=False,

                covered_by_contract_ids=[],

                notes=[
                    (
                        "The user explicitly requested a share "
                        "of the total. This is a derived output "
                        "requirement and is not satisfied merely "
                        "by binding the underlying source metric."
                    )
                ],
            )
        )


    # --------------------------------------------------------
    # EXPLICIT PHYSICAL COLUMN REFERENCES
    # --------------------------------------------------------
    #
    # This keeps the guard useful beyond the small semantic
    # vocabulary above.
    #
    # If a real catalog column is literally named by the user,
    # it must survive into at least one validated contract.
    # --------------------------------------------------------

    for column_name in (
        catalog_columns
    ):
        normalized_column = (
            normalize_text(
                column_name
            )
        )

        if not normalized_column:
            continue

        if (
            normalized_column
            in
            semantic_column_names
        ):
            continue

        if not contains_phrase(
            normalized_objective,
            normalized_column,
        ):
            continue

        requirement_id = (
            "column:"
            +
            normalized_column.replace(
                " ",
                "_",
            )
        )

        requirements.append(
            ObjectiveCoverageRequirement(
                requirement_id=
                    requirement_id,

                concept=
                    column_name,

                requirement_type=
                    "column",

                requested_phrases=[
                    normalized_column
                ],

                candidate_columns=[
                    column_name
                ],

                allowed_roles=[],

                required_aggregation=None,

                covered=False,

                covered_by_contract_ids=[],

                notes=[],
            )
        )


    # --------------------------------------------------------
    # DEDUPLICATE
    # --------------------------------------------------------

    unique: dict[
        str,
        ObjectiveCoverageRequirement
    ] = {}

    for requirement in (
        requirements
    ):
        unique[
            requirement.requirement_id
        ] = (
            requirement
        )

    return list(
        unique.values()
    )


# ============================================================
# CONTRACT COVERAGE
# ============================================================

def contract_covers_requirement(
    *,
    contract: AnalyticalContract,

    requirement: (
        ObjectiveCoverageRequirement
    ),
) -> bool:
    if (
        contract.status
        !=
        "validated"
    ):
        return False

    # --------------------------------------------------------
    # DERIVED OUTPUT REQUIREMENTS
    # DATALENS_OBJECTIVE_COVERAGE_DERIVED_SHARE_SATISFACTION_V0_1
    # --------------------------------------------------------
    #
    # Derived outputs deliberately have no physical candidate
    # columns. They therefore must be evaluated before the
    # physical-column coverage path below.
    #
    # v0.1 supports only the canonical share_of_total semantic.
    # Unknown/future derived metrics remain fail-closed.
    # --------------------------------------------------------

    if (
        requirement.requirement_type
        ==
        "derived_metric"
    ):
        if (
            requirement.concept
            !=
            "share_of_total"
        ):
            return False


        share_spec = (
            contract.share_of_total
        )


        if (
            share_spec
            is None
        ):
            return False


        return (
            share_spec.reference
            ==
            "sum_of_group_values"
        )


    candidate_names = {
        normalize_text(
            column_name
        )

        for column_name
        in requirement.candidate_columns
    }

    if not candidate_names:
        return False

    matching_bindings = [
        binding

        for binding
        in contract.bindings

        if (
            normalize_text(
                binding.column
            )
            in
            candidate_names

            or
            (
                requirement.requirement_type
                ==
                "column"
                and
                bool(
                    binding.semantic_concept
                )
                and
                normalize_text(
                    str(
                        binding.semantic_concept
                    )
                )
                in
                candidate_names
            )
        )
    ]

    if not matching_bindings:
        return False


    # --------------------------------------------------------
    # EXACT PHYSICAL COLUMN
    # --------------------------------------------------------

    if (
        requirement.requirement_type
        ==
        "column"
    ):
        return True


    # --------------------------------------------------------
    # ROLE FIDELITY
    # --------------------------------------------------------

    allowed_roles = set(
        requirement.allowed_roles
    )

    role_matches = [
        binding

        for binding
        in matching_bindings

        if (
            not allowed_roles
            or
            binding.role
            in
            allowed_roles
        )
    ]

    if not role_matches:
        return False


    # --------------------------------------------------------
    # DIMENSION
    # --------------------------------------------------------

    if (
        requirement.requirement_type
        ==
        "dimension"
    ):
        return True


    # --------------------------------------------------------
    # METRIC AGGREGATION
    # --------------------------------------------------------

    required_aggregation = (
        requirement
        .required_aggregation
    )

    if required_aggregation is None:
        return True

    aggregation = (
        contract.aggregation
    )

    if aggregation is None:
        return False

    if (
        aggregation.function
        !=
        required_aggregation
    ):
        return False

    source_role = (
        aggregation.source_role
    )

    if source_role is None:
        return False

    return any(
        binding.role
        ==
        source_role

        for binding
        in role_matches
    )


# ============================================================
# SHARED GROUPING TOPOLOGY
# ============================================================

def first_phrase_position(
    *,
    objective: str,
    phrases: list[
        str
    ],
) -> tuple[
    int,
    int,
] | None:
    normalized_objective = (
        normalize_text(
            objective
        )
    )

    matches: list[
        tuple[
            int,
            int,
        ]
    ] = []


    for phrase in phrases:
        normalized_phrase = (
            normalize_text(
                phrase
            )
        )

        if not normalized_phrase:
            continue

        position = (
            normalized_objective.find(
                normalized_phrase
            )
        )

        if position < 0:
            continue

        matches.append(
            (
                position,
                position
                +
                len(
                    normalized_phrase
                ),
            )
        )


    if not matches:
        return None


    return min(
        matches,
        key=lambda value: (
            value[
                0
            ]
        ),
    )


def shared_metric_dimension_scope(
    *,
    objective: str,

    metric_requirements: list[
        ObjectiveCoverageRequirement
    ],

    dimension_requirements: list[
        ObjectiveCoverageRequirement
    ],
) -> bool:
    """
    Detect only the conservative shared-grouping form:

        metric A and metric B
            BY
        dimension X and dimension Y

    Example accepted:

        revenue and return rate by region and channel

    Example deliberately NOT expanded:

        revenue by region and return rate by channel

    This avoids inventing cross-metric topology when the request
    scopes dimensions independently.
    """

    if (
        not metric_requirements
        or
        not dimension_requirements
    ):
        return False


    metric_positions = [
        first_phrase_position(
            objective=
                objective,

            phrases=
                requirement
                .requested_phrases,
        )

        for requirement
        in metric_requirements
    ]


    dimension_positions = [
        first_phrase_position(
            objective=
                objective,

            phrases=
                requirement
                .requested_phrases,
        )

        for requirement
        in dimension_requirements
    ]


    if any(
        value is None

        for value
        in (
            *
            metric_positions,
            *
            dimension_positions,
        )
    ):
        return False


    concrete_metric_positions = [
        value

        for value
        in metric_positions

        if value is not None
    ]

    concrete_dimension_positions = [
        value

        for value
        in dimension_positions

        if value is not None
    ]


    metric_end = max(
        value[
            1
        ]

        for value
        in concrete_metric_positions
    )


    dimension_start = min(
        value[
            0
        ]

        for value
        in concrete_dimension_positions
    )


    if (
        metric_end
        >=
        dimension_start
    ):
        return False


    normalized_objective = (
        normalize_text(
            objective
        )
    )


    bridge = (
        normalized_objective[
            metric_end:
            dimension_start
        ]
    )


    return (
        re.search(
            r"\b(?:par|by|selon)\b",
            bridge,
        )
        is not None
    )


def contract_groups_by_requirement(
    *,
    contract: AnalyticalContract,

    requirement: ObjectiveCoverageRequirement,
) -> bool:
    candidate_names = {
        normalize_text(
            column_name
        )

        for column_name
        in requirement.candidate_columns
    }


    matching_bindings = [
        binding

        for binding
        in contract.bindings

        if (
            normalize_text(
                binding.column
            )
            in
            candidate_names
        )
    ]


    if not matching_bindings:
        return False


    aggregation = (
        contract.aggregation
    )


    if aggregation is None:
        return False


    grouped_roles = set(
        aggregation.group_by_roles
    )


    return any(
        binding.role
        in
        grouped_roles

        for binding
        in matching_bindings
    )


def contract_groups_by_single_requirement(
    *,
    contract: AnalyticalContract,

    requirement: ObjectiveCoverageRequirement,
) -> bool:
    """
    Require one marginal grouping dimension.

    For a request such as:

        revenue by region and by channel

    DataLens expects independent region and channel analyses.

    A region+channel joint cross-tab is not equivalent to either
    marginal aggregation and therefore does not satisfy this
    requirement.
    """

    aggregation = (
        contract.aggregation
    )


    if aggregation is None:
        return False


    candidate_names = {
        normalize_text(
            column_name
        )

        for column_name
        in requirement.candidate_columns
    }


    matching_roles = {
        binding.role

        for binding
        in contract.bindings

        if (
            normalize_text(
                binding.column
            )
            in
            candidate_names
        )
    }


    if not matching_roles:
        return False


    grouped_roles = list(
        aggregation.group_by_roles
    )


    if (
        len(
            grouped_roles
        )
        !=
        1
    ):
        return False


    return (
        grouped_roles[
            0
        ]
        in
        matching_roles
    )


# ============================================================
# ANALYTICAL INTENT COVERAGE
# DATALENS_OBJECTIVE_INTENT_COVERAGE_V0_1
# ============================================================

def _intent_binding_matches_metric(
    *,
    contract: AnalyticalContract,
    intent: ObjectiveCoverageIntentRequirement,
) -> bool:
    aggregation = (
        contract.aggregation
    )

    if aggregation is None:
        return False

    if (
        intent.required_aggregation
        is not None
        and
        aggregation.function
        !=
        intent.required_aggregation
    ):
        return False

    source_role = (
        aggregation.source_role
    )

    if source_role is None:
        return False

    candidates = {
        normalize_text(
            column
        )

        for column
        in intent.candidate_metric_columns
    }

    metric_concept = (
        normalize_text(
            intent.metric_concept
        )
    )

    for binding in contract.bindings:
        if (
            binding.role
            !=
            source_role
        ):
            continue

        if (
            normalize_text(
                binding.column
            )
            in candidates
        ):
            return True

        semantic_concept = (
            normalize_text(
                str(
                    binding.semantic_concept
                    or
                    ""
                )
            )
        )

        if (
            semantic_concept
            and
            semantic_concept
            ==
            metric_concept
        ):
            return True

    return False


def _intent_group_matches(
    *,
    contract: AnalyticalContract,
    concepts: set[str],
) -> bool:
    aggregation = (
        contract.aggregation
    )

    if aggregation is None:
        return False

    grouped_roles = set(
        aggregation.group_by_roles
    )

    if not grouped_roles:
        return False

    for binding in contract.bindings:
        if (
            binding.role
            not in
            grouped_roles
        ):
            continue

        binding_text = normalize_text(
            binding.column
        )

        semantic_text = normalize_text(
            str(
                binding.semantic_concept
                or
                ""
            )
        )

        tokens = {
            *binding_text.split(),
            *semantic_text.split(),
        }

        if (
            tokens
            &
            concepts
        ):
            return True

    return False


def contract_covers_intent(
    *,
    contract: AnalyticalContract,
    intent: ObjectiveCoverageIntentRequirement,
) -> bool:
    if (
        contract.status
        !=
        "validated"
    ):
        return False

    if not _intent_binding_matches_metric(
        contract=contract,
        intent=intent,
    ):
        return False

    if (
        intent.intent_kind
        ==
        "scalar_total"
    ):
        if (
            contract.family
            not in {
                "aggregation",
                "descriptive_metric",
            }
        ):
            return False

        aggregation = (
            contract.aggregation
        )

        if aggregation is None:
            return False

        return (
            len(
                aggregation.group_by_roles
            )
            ==
            0
        )

    if (
        intent.intent_kind
        ==
        "monthly_time_series"
    ):
        if (
            contract.family
            !=
            "time_series"
        ):
            return False

        grain = normalize_text(
            str(
                contract.analytical_grain
                or
                ""
            )
        )

        time_bindings = [
            binding
            for binding
            in contract.bindings
            if binding.role == "time"
        ]

        monthly_tokens = {
            "month",
            "monthly",
            "mois",
            "mensuel",
            "mensuelle",
        }

        return (
            grain
            in monthly_tokens
            or
            any(
                bool(
                    set(
                        normalize_text(
                            binding.column
                        ).split()
                    )
                    &
                    monthly_tokens
                )
                for binding
                in time_bindings
            )
        )

    if (
        intent.intent_kind
        ==
        "categorical_breakdown"
    ):
        if (
            contract.family
            not in {
                "aggregation",
                "group_comparison",
            }
        ):
            return False

        return _intent_group_matches(
            contract=contract,
            concepts={
                "category",
                "categorie",
                "categories",
                "categ",
            },
        )

    if (
        intent.intent_kind
        ==
        "entity_ranking"
    ):
        if (
            contract.family
            !=
            "ranking"
        ):
            return False

        ranking = (
            contract.ranking
        )

        if ranking is None:
            return False

        if (
            intent.ranking_order
            is not None
            and
            ranking.order
            !=
            intent.ranking_order
        ):
            return False

        if (
            intent.ranking_limit
            is not None
            and
            ranking.limit
            !=
            intent.ranking_limit
        ):
            return False

        return _intent_group_matches(
            contract=contract,
            concepts={
                "client",
                "clients",
                "customer",
                "customers",
                "buyer",
                "buyers",
                "acheteur",
                "acheteurs",
                "user",
                "users",
            },
        )

    return False


def _explicit_entity_ranking_limit(
    objective: str,
) -> int | None:
    normalized = (
        normalize_text(
            objective
        )
    )

    patterns = (
        r"\btop\s+(\d{1,3})\b",
        (
            r"\b(\d{1,3})\s+"
            r"(?:clients?|customers?|buyers?|acheteurs?|users?)\b"
        ),
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            normalized,
        )

        if match is None:
            continue

        value = int(
            match.group(
                1
            )
        )

        if (
            1
            <=
            value
            <=
            100
        ):
            return value

    return None


def build_intent_requirements(
    *,
    objective: str,
    requirements: list[
        ObjectiveCoverageRequirement
    ],
    contracts: list[
        AnalyticalContract
    ],
) -> list[
    ObjectiveCoverageIntentRequirement
]:
    metric_requirements = [
        requirement

        for requirement
        in requirements

        if (
            requirement.requirement_type
            ==
            "metric"
        )
    ]

    # Fail closed to the existing Objective Coverage logic for
    # zero-metric and multi-metric objectives. v0.1 deliberately
    # addresses the same-metric / multi-intent gap only.
    if (
        len(
            metric_requirements
        )
        !=
        1
    ):
        return []

    metric = (
        metric_requirements[
            0
        ]
    )

    if not metric.candidate_columns:
        return []

    normalized = (
        normalize_text(
            objective
        )
    )

    detected: list[
        ObjectiveCoverageIntentRequirement
    ] = []

    metric_kwargs = {
        "metric_concept":
            metric.concept,

        "candidate_metric_columns":
            list(
                metric.candidate_columns
            ),

        "required_aggregation":
            metric.required_aggregation,
    }

    # --------------------------------------------------------
    # EXPLICIT SCALAR TOTAL
    # --------------------------------------------------------

    if re.search(
        r"\btotal(?:e|es|s)?\b",
        normalized,
    ):
        detected.append(
            ObjectiveCoverageIntentRequirement(
                intent_id=(
                    "intent:"
                    + metric.concept
                    + ":scalar_total"
                ),
                intent_kind=
                    "scalar_total",
                requested_phrases=[
                    "total"
                ],
                required_family=
                    "aggregation",
                notes=[
                    (
                        "An explicit total requires an "
                        "ungrouped scalar aggregation."
                    )
                ],
                **metric_kwargs,
            )
        )

    # --------------------------------------------------------
    # EXPLICIT MONTHLY SERIES
    # --------------------------------------------------------

    monthly_requested = bool(
        re.search(
            (
                r"\b(?:mensuel|mensuelle|mensuels|mensuelles|monthly)\b"
                r"|\bpar\s+mois\b"
                r"|\bchaque\s+mois\b"
            ),
            normalized,
        )
    )

    if monthly_requested:
        detected.append(
            ObjectiveCoverageIntentRequirement(
                intent_id=(
                    "intent:"
                    + metric.concept
                    + ":monthly_time_series"
                ),
                intent_kind=
                    "monthly_time_series",
                requested_phrases=[
                    "monthly"
                ],
                required_family=
                    "time_series",
                required_grain=
                    "month",
                notes=[
                    (
                        "An explicit monthly request requires "
                        "a validated monthly time-series contract."
                    )
                ],
                **metric_kwargs,
            )
        )

    # --------------------------------------------------------
    # EXPLICIT CATEGORY BREAKDOWN
    # --------------------------------------------------------

    category_requested = (
        bool(
            re.search(
                r"\b(?:categorie|categories|category|categ)\b",
                normalized,
            )
        )
        and
        bool(
            re.search(
                r"\b(?:par|by|selon)\b",
                normalized,
            )
        )
    )

    if category_requested:
        detected.append(
            ObjectiveCoverageIntentRequirement(
                intent_id=(
                    "intent:"
                    + metric.concept
                    + ":category_breakdown"
                ),
                intent_kind=
                    "categorical_breakdown",
                requested_phrases=[
                    "category breakdown"
                ],
                required_family=
                    "aggregation",
                required_dimension_concepts=[
                    "category"
                ],
                notes=[
                    (
                        "An explicit category breakdown requires "
                        "a validated grouped metric contract."
                    )
                ],
                **metric_kwargs,
            )
        )

    # --------------------------------------------------------
    # EXPLICIT ENTITY TOP-N
    # --------------------------------------------------------

    ranking_limit = (
        _explicit_entity_ranking_limit(
            objective
        )
    )

    entity_requested = bool(
        re.search(
            (
                r"\b(?:client|clients|customer|customers|"
                r"buyer|buyers|acheteur|acheteurs|user|users)\b"
            ),
            normalized,
        )
    )

    descending_requested = bool(
        re.search(
            (
                r"\btop\b"
                r"|\ble\s+plus\b"
                r"|\bplus\s+de\b"
                r"|\bhighest\b"
                r"|\bmost\b"
                r"|\bmeilleur(?:s)?\b"
            ),
            normalized,
        )
    )

    if (
        ranking_limit
        is not None
        and
        entity_requested
        and
        descending_requested
    ):
        detected.append(
            ObjectiveCoverageIntentRequirement(
                intent_id=(
                    "intent:"
                    + metric.concept
                    + ":entity_ranking:"
                    + str(
                        ranking_limit
                    )
                ),
                intent_kind=
                    "entity_ranking",
                requested_phrases=[
                    (
                        "top "
                        + str(
                            ranking_limit
                        )
                    )
                ],
                required_family=
                    "ranking",
                required_dimension_concepts=[
                    "customer"
                ],
                ranking_order=
                    "descending",
                ranking_limit=
                    ranking_limit,
                notes=[
                    (
                        "An explicit Top-N entity request requires "
                        "a validated descending ranking with the "
                        "same limit."
                    )
                ],
                **metric_kwargs,
            )
        )

    evaluated: list[
        ObjectiveCoverageIntentRequirement
    ] = []

    for intent in detected:
        covering_contract_ids = [
            contract.contract_id

            for contract
            in contracts

            if contract_covers_intent(
                contract=contract,
                intent=intent,
            )
        ]

        evaluated.append(
            intent.model_copy(
                update={
                    "covered":
                        bool(
                            covering_contract_ids
                        ),

                    "covered_by_contract_ids":
                        covering_contract_ids,
                }
            )
        )

    return evaluated


def build_topology_requirements(
    *,
    objective: str,

    requirements: list[
        ObjectiveCoverageRequirement
    ],

    contracts: list[
        AnalyticalContract
    ],
) -> list[
    ObjectiveCoverageTopologyRequirement
]:
    # DATALENS_OBJECTIVE_COVERAGE_PAIRWISE_TOPOLOGY_V0_3
    #
    # Shared metric/dimension wording creates atomic
    # metric-by-dimension requirements.
    #
    # Example:
    #
    #     revenue and return rate
    #     by region and by channel
    #
    # becomes:
    #
    #     revenue x region
    #     revenue x channel
    #     return_rate x region
    #     return_rate x channel
    #
    # These requirements may be satisfied by separate validated
    # contracts. This matches the marginal comparisons users
    # normally request and avoids silently replacing them with a
    # region x channel cross-tab.

    metric_requirements = [
        requirement

        for requirement
        in requirements

        if (
            requirement.requirement_type
            ==
            "metric"
        )
    ]


    dimension_requirements = [
        requirement

        for requirement
        in requirements

        if (
            requirement.requirement_type
            ==
            "dimension"
        )
    ]


    if not (
        shared_metric_dimension_scope(
            objective=
                objective,

            metric_requirements=
                metric_requirements,

            dimension_requirements=
                dimension_requirements,
        )
    ):
        return []


    topology_requirements: list[
        ObjectiveCoverageTopologyRequirement
    ] = []


    for metric_requirement in (
        metric_requirements
    ):
        for dimension_requirement in (
            dimension_requirements
        ):

            covering_contract_ids: list[
                str
            ] = []


            for contract in contracts:

                if not (
                    contract_covers_requirement(
                        contract=
                            contract,

                        requirement=
                            metric_requirement,
                    )
                ):
                    continue


                if not (
                    contract_groups_by_single_requirement(
                        contract=
                            contract,

                        requirement=
                            dimension_requirement,
                    )
                ):
                    continue


                covering_contract_ids.append(
                    contract.contract_id
                )


            topology_requirements.append(
                ObjectiveCoverageTopologyRequirement(
                    topology_id=(
                        "topology:"
                        +
                        metric_requirement.concept
                        +
                        ":by:"
                        +
                        dimension_requirement.concept
                    ),

                    metric_concept=
                        metric_requirement.concept,

                    required_dimension_concepts=[
                        dimension_requirement.concept
                    ],

                    covered=
                        bool(
                            covering_contract_ids
                        ),

                    covered_by_contract_ids=
                        covering_contract_ids,

                    notes=[
                        (
                            "The requested metric/dimension "
                            "marginal must appear in at least "
                            "one validated analytical contract."
                        )
                    ],
                )
            )


    return (
        topology_requirements
    )


# ============================================================
# PUBLIC REPORT
# ============================================================

def build_objective_coverage(
    *,
    objective: str,
    catalog: Any,
    contracts: list[
        AnalyticalContract
    ],
) -> ObjectiveCoverageReport:
    requirements = (
        extract_objective_requirements(
            objective=
                objective,

            catalog=
                catalog,
        )
    )

    if not requirements:
        return (
            ObjectiveCoverageReport(
                status=
                    "not_applicable",

                requirement_count=0,

                covered_count=0,

                missing_count=0,

                requirements=[],

                topology_requirement_count=0,

                topology_covered_count=0,

                topology_missing_count=0,

                topology_requirements=[],

                notes=[
                    (
                        "No conservative deterministic "
                        "objective requirement was extracted."
                    )
                ],
            )
        )


    evaluated: list[
        ObjectiveCoverageRequirement
    ] = []

    for requirement in (
        requirements
    ):
        covering_contract_ids = [
            contract.contract_id

            for contract
            in contracts

            if contract_covers_requirement(
                contract=
                    contract,

                requirement=
                    requirement,
            )
        ]

        evaluated.append(
            requirement.model_copy(
                update={
                    "covered":
                        bool(
                            covering_contract_ids
                        ),

                    "covered_by_contract_ids":
                        covering_contract_ids,
                }
            )
        )


    covered_count = sum(
        1

        for requirement
        in evaluated

        if requirement.covered
    )

    missing_count = (
        len(
            evaluated
        )
        -
        covered_count
    )


    topology_requirements = (
        build_topology_requirements(
            objective=
                objective,

            requirements=
                evaluated,

            contracts=
                contracts,
        )
    )


    topology_covered_count = sum(
        1

        for topology
        in topology_requirements

        if topology.covered
    )


    topology_missing_count = (
        len(
            topology_requirements
        )
        -
        topology_covered_count
    )


    intent_requirements = (
        build_intent_requirements(
            objective=
                objective,

            requirements=
                evaluated,

            contracts=
                contracts,
        )
    )


    intent_covered_count = sum(
        1

        for intent
        in intent_requirements

        if intent.covered
    )


    intent_missing_count = (
        len(
            intent_requirements
        )
        -
        intent_covered_count
    )


    status: ObjectiveCoverageStatus = (
        "complete"

        if (
            missing_count
            ==
            0
            and
            topology_missing_count
            ==
            0
            and
            intent_missing_count
            ==
            0
        )

        else
        "incomplete"
    )


    notes = [
        (
            "Coverage is evaluated across the UNION "
            "of all validated analytical contracts."
        ),
        (
            "A technically valid contract is not enough: "
            "explicit requested metrics and dimensions "
            "must also be preserved."
        ),
    ]


    return (
        ObjectiveCoverageReport(
            status=
                status,

            requirement_count=
                len(
                    evaluated
                ),

            covered_count=
                covered_count,

            missing_count=
                missing_count,

            requirements=
                evaluated,

            topology_requirement_count=
                len(
                    topology_requirements
                ),

            topology_covered_count=
                topology_covered_count,

            topology_missing_count=
                topology_missing_count,

            topology_requirements=
                topology_requirements,

            intent_requirement_count=
                len(
                    intent_requirements
                ),

            intent_covered_count=
                intent_covered_count,

            intent_missing_count=
                intent_missing_count,

            intent_requirements=
                intent_requirements,

            notes=
                notes,
        )
    )


# ============================================================
# FAIL-CLOSED PLANNER GUARD
# DATALENS_OBJECTIVE_COVERAGE_FAIL_CLOSED_GUARD_V0_1
# ============================================================

class ObjectiveCoverageIncompleteError(
    ValueError
):
    """
    Raised when a technically validated AI planner report does
    not preserve all deterministic requirements extracted from
    the user objective.

    This is deliberately distinct from Request Coverage:

    - Request Coverage proves that the request itself was not
      lost across the documentary planner boundary.
    - Objective Coverage proves that the validated analytical
      contracts actually preserve the explicit metrics and
      dimensions required by the user objective.
    """

    def __init__(
        self,
        report: ObjectiveCoverageReport,
    ) -> None:
        self.report = (
            report
        )

        missing_concepts = [
            requirement.concept

            for requirement
            in report.requirements

            if not requirement.covered
        ]

        suffix = (
            ", ".join(
                missing_concepts
            )
        )

        topology_suffix = ", ".join(
            (
                topology.metric_concept
                +
                " by "
                +
                "+".join(
                    topology
                    .required_dimension_concepts
                )
            )

            for topology
            in report.topology_requirements

            if not topology.covered
        )


        super().__init__(
            (
                "Objective coverage is incomplete."
                +
                (
                    " Missing concepts: "
                    + suffix
                    + "."

                    if suffix
                    else
                    ""
                )
                +
                (
                    " Missing topology: "
                    + topology_suffix
                    + "."

                    if topology_suffix
                    else
                    ""
                )
            )
        )


def validated_contracts_from_planner_report(
    planner_report: Any,
) -> list[
    AnalyticalContract
]:
    """
    Extract only contracts that passed Python validation.

    Blocked, ambiguous, rejected, malformed or contract-less
    planner items never contribute to semantic coverage.
    """

    contracts: list[
        AnalyticalContract
    ] = []


    for item in (
        getattr(
            planner_report,
            "items",
            [],
        )
        or []
    ):
        validation_status = (
            getattr(
                item,
                "validation_status",
                None,
            )
        )

        if (
            validation_status
            !=
            "validated"
        ):
            continue


        contract = (
            getattr(
                item,
                "contract",
                None,
            )
        )


        if contract is None:
            continue


        if not isinstance(
            contract,
            AnalyticalContract,
        ):
            contract = (
                AnalyticalContract
                .model_validate(
                    contract
                )
            )


        if (
            contract.status
            !=
            "validated"
        ):
            continue


        contracts.append(
            contract
        )


    return (
        contracts
    )


def require_objective_coverage(
    *,
    objective: str,
    catalog: Any,
    planner_report: Any,
) -> ObjectiveCoverageReport:
    """
    Fail closed before analytical tool execution when explicit
    deterministic objective requirements are not covered by the
    union of Python-validated analytical contracts.

    NOT_APPLICABLE remains executable because the conservative
    guard intentionally abstains when it cannot deterministically
    extract an objective requirement.
    """

    contracts = (
        validated_contracts_from_planner_report(
            planner_report
        )
    )


    report = (
        build_objective_coverage(
            objective=
                objective,

            catalog=
                catalog,

            contracts=
                contracts,
        )
    )


    if (
        report.status
        ==
        "incomplete"
    ):
        raise (
            ObjectiveCoverageIncompleteError(
                report
            )
        )


    return (
        report
    )
