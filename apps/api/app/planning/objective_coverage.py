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


def resolve_candidate_columns(
    *,
    catalog_columns: list[
        str
    ],

    candidate_names: tuple[
        str,
        ...
    ],
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

        if actual is None:
            continue

        resolved.append(
            actual
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
                    "but no compatible physical column "
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
