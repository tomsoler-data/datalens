from __future__ import annotations


from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.planning.analytical_contract import (
    AggregationSpec,
    AnalyticalContract,
    ContractProvenance,
    DerivedVariableSpec,
    RankingSpec,
    VariableBinding,
    WindowSpec,
)

from app.planning.schemas import (
    RequestedAnalysisPlan,
    RequestedAnalysisPlanReport,
    RequestedColumnMatch,
)


# ============================================================
# VERSION
# ============================================================

LEGACY_CONTRACT_ADAPTER_RULE_VERSION = (
    "legacy_contract_adapter_v0.1"
)


# ============================================================
# MIGRATION REPORT
# ============================================================

LegacyMappingStatus = Literal[
    "mapped",
    "blocked",
    "ambiguous",
]


class LegacyContractMapping(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    request_id: str

    legacy_kind: str

    mapping_status: (
        LegacyMappingStatus
    )

    contract: AnalyticalContract

    adapter_notes: list[
        str
    ] = Field(
        default_factory=list
    )


class LegacyContractMigrationReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    request_count: int

    mapped_count: int

    blocked_count: int

    ambiguous_count: int

    mappings: list[
        LegacyContractMapping
    ]

    adapter_rule_version: str = (
        LEGACY_CONTRACT_ADAPTER_RULE_VERSION
    )


# ============================================================
# HELPERS
# ============================================================

def find_match(
    plan: RequestedAnalysisPlan,
    *concepts: str,
) -> RequestedColumnMatch | None:
    wanted = {
        concept.casefold()
        for concept
        in concepts
    }


    for match in (
        plan.matched_columns
    ):
        if (
            match.concept
            .casefold()
            in wanted
        ):
            return match


    return None


def physical_binding(
    plan: RequestedAnalysisPlan,
    *,
    role: str,
    concepts: tuple[
        str,
        ...
    ],
    semantic_concept: (
        str
        | None
    ) = None,
) -> VariableBinding | None:
    match = find_match(
        plan,
        *concepts,
    )


    if (
        match is None
    ):
        return None


    return VariableBinding(
        role=role,  # type: ignore[arg-type]
        column=match.column,
        dataset_id=(
            match.dataset_id
        ),
        dataset_filename=(
            match.dataset_filename
        ),
        semantic_concept=(
            semantic_concept
            or match.concept
        ),
        analysis_kind=(
            match.analysis_kind
        ),
    )


def derived_binding(
    *,
    role: str,
    column: str,
    semantic_concept: str,
) -> VariableBinding:
    return VariableBinding(
        role=role,  # type: ignore[arg-type]
        column=column,
        semantic_concept=(
            semantic_concept
        ),
        analysis_kind=(
            "derived"
        ),
    )


def without_none(
    values: list[
        VariableBinding
        | None
    ],
) -> list[
    VariableBinding
]:
    return [
        value
        for value
        in values
        if value is not None
    ]


def provenance_from_plan(
    plan: RequestedAnalysisPlan,
) -> ContractProvenance:
    return ContractProvenance(
        source_filename=(
            plan.source_filename
        ),
        source_locator=(
            plan.source_locator
        ),
        page_number=(
            plan.page_number
        ),
        source_chunk_id=(
            plan.source_chunk_id
        ),
        evidence_unit_id=(
            plan.evidence_unit_id
        ),
        evidence_quote=(
            plan.evidence_quote
        ),
    )


def contract_status_from_plan(
    plan: RequestedAnalysisPlan,
) -> str:
    if (
        plan.status ==
        "ready"
    ):
        return "validated"


    return plan.status


def base_contract_kwargs(
    plan: RequestedAnalysisPlan,
) -> dict:
    return {
        "contract_id":
            (
                "legacy:"
                f"{plan.request_id}"
            ),

        "origin":
            "legacy_adapter",

        "status":
            contract_status_from_plan(
                plan
            ),

        "title":
            plan.request_text,

        "request_text":
            plan.request_text,

        "required_dataset_ids":
            list(
                plan.required_dataset_ids
            ),

        "required_dataset_filenames":
            list(
                plan.required_dataset_filenames
            ),

        "required_operations":
            list(
                plan.required_operations
            ),

        "provenance":
            provenance_from_plan(
                plan
            ),

        "reasons":
            [
                *plan.reasons,
                (
                    "Contrat généré en parallèle par "
                    "l'adaptateur de migration. "
                    "L'ancien planner reste la source "
                    "de résolution pendant cette étape."
                ),
            ],

        "blockers":
            list(
                plan.blockers
            ),
    }


def age_derived_variables(
    plan: RequestedAnalysisPlan,
) -> list[
    DerivedVariableSpec
]:
    birth = find_match(
        plan,
        "birth",
        "date_of_birth",
    )

    time = find_match(
        plan,
        "time",
        "event_time",
        "date",
    )

    customer = find_match(
        plan,
        "customer_id",
        "entity_id",
    )


    source_columns = [
        match.column
        for match
        in [
            birth,
            time,
        ]
        if match is not None
    ]


    group_columns = (
        [
            customer.column
        ]
        if customer is not None
        else []
    )


    return [
        DerivedVariableSpec(
            output_column=(
                "age_at_first_purchase"
            ),
            operation=(
                "age_at_event"
            ),
            source_columns=(
                source_columns
            ),
            group_by_columns=(
                group_columns
            ),
            parameters={
                "event_selector":
                    "first",
            },
            rationale=(
                "Convertir une date de naissance et "
                "une date d'événement en âge au grain "
                "analytique, sans utiliser le LLM pour "
                "le calcul."
            ),
        ),
    ]


# ============================================================
# LEGACY → GENERIC CONTRACT
# ============================================================

def adapt_requested_plan(
    plan: RequestedAnalysisPlan,
) -> LegacyContractMapping:
    kind = (
        plan.kind
    )


    base = (
        base_contract_kwargs(
            plan
        )
    )


    customer = (
        physical_binding(
            plan,
            role="entity",
            concepts=(
                "customer_id",
                "client_id",
            ),
            semantic_concept=(
                "entity_identifier"
            ),
        )
    )

    product = (
        physical_binding(
            plan,
            role="entity",
            concepts=(
                "product_id",
                "item_id",
            ),
            semantic_concept=(
                "entity_identifier"
            ),
        )
    )

    amount = (
        physical_binding(
            plan,
            role="value",
            concepts=(
                "amount",
                "revenue",
                "price",
            ),
            semantic_concept=(
                "monetary_measure"
            ),
        )
    )

    time = (
        physical_binding(
            plan,
            role="time",
            concepts=(
                "time",
                "event_time",
                "date",
            ),
            semantic_concept=(
                "event_timestamp"
            ),
        )
    )

    category_group = (
        physical_binding(
            plan,
            role="group",
            concepts=(
                "category",
                "product_category",
            ),
            semantic_concept=(
                "categorical_dimension"
            ),
        )
    )

    category_dimension = (
        physical_binding(
            plan,
            role="dimension",
            concepts=(
                "category",
                "product_category",
            ),
            semantic_concept=(
                "categorical_dimension"
            ),
        )
    )

    gender_x = (
        physical_binding(
            plan,
            role="x",
            concepts=(
                "gender",
                "sex",
            ),
            semantic_concept=(
                "categorical_attribute"
            ),
        )
    )

    category_y = (
        physical_binding(
            plan,
            role="y",
            concepts=(
                "category",
                "product_category",
            ),
            semantic_concept=(
                "categorical_dimension"
            ),
        )
    )


    # --------------------------------------------------------
    # TIME SERIES
    # --------------------------------------------------------

    if (
        kind ==
        "revenue_moving_average"
    ):
        contract = AnalyticalContract(
            **base,
            family="time_series",
            analytical_grain="period",
            bindings=without_none(
                [
                    time,
                    amount,
                ]
            ),
            aggregation=(
                AggregationSpec(
                    function="sum",
                    source_role="value",
                    group_by_roles=[
                        "time",
                    ],
                    output_name=(
                        "aggregated_value"
                    ),
                )
            ),
            window=(
                WindowSpec(
                    operation=(
                        "moving_average"
                    ),
                    window=3,
                    minimum_periods=1,
                )
            ),
        )


        return mapped(
            plan,
            contract,
            notes=[
                (
                    "La fenêtre 3 conserve le comportement "
                    "par défaut de l'exécuteur legacy. "
                    "Le futur AI Planner devra l'exposer "
                    "explicitement dans son contrat."
                ),
            ],
        )


    if (
        kind ==
        "customers_by_period"
    ):
        contract = AnalyticalContract(
            **base,
            family="time_series",
            analytical_grain="period",
            bindings=without_none(
                [
                    time,
                    customer,
                ]
            ),
            aggregation=(
                AggregationSpec(
                    function=(
                        "distinct_count"
                    ),
                    source_role=(
                        "entity"
                    ),
                    group_by_roles=[
                        "time",
                    ],
                    output_name=(
                        "distinct_entity_count"
                    ),
                )
            ),
        )


        return mapped(
            plan,
            contract,
        )


    # --------------------------------------------------------
    # AGGREGATION / METRICS
    # --------------------------------------------------------

    if (
        kind ==
        "revenue_by_category"
    ):
        contract = AnalyticalContract(
            **base,
            family="aggregation",
            analytical_grain="group",
            bindings=without_none(
                [
                    category_group,
                    amount,
                ]
            ),
            aggregation=(
                AggregationSpec(
                    function="sum",
                    source_role="value",
                    group_by_roles=[
                        "group",
                    ],
                    output_name=(
                        "aggregated_value"
                    ),
                )
            ),
        )


        return mapped(
            plan,
            contract,
        )


    if (
        kind ==
        "transaction_count"
    ):
        contract = AnalyticalContract(
            **base,
            family=(
                "descriptive_metric"
            ),
            analytical_grain="dataset",
            bindings=[],
            aggregation=(
                AggregationSpec(
                    function="count",
                    source_role=None,
                    output_name=(
                        "row_count"
                    ),
                )
            ),
        )


        return mapped(
            plan,
            contract,
        )


    if (
        kind ==
        "products_sold_count"
    ):
        contract = AnalyticalContract(
            **base,
            family=(
                "descriptive_metric"
            ),
            analytical_grain="dataset",
            bindings=without_none(
                [
                    product,
                ]
            ),
            aggregation=(
                AggregationSpec(
                    function="count",
                    source_role=(
                        "entity"
                    ),
                    output_name=(
                        "entity_occurrence_count"
                    ),
                )
            ),
        )


        return mapped(
            plan,
            contract,
        )


    if (
        kind ==
        "product_category_distribution"
    ):
        contract = AnalyticalContract(
            **base,
            family="aggregation",
            analytical_grain="group",
            bindings=without_none(
                [
                    category_dimension,
                    product,
                ]
            ),
            aggregation=(
                AggregationSpec(
                    function=(
                        "distinct_count"
                    ),
                    source_role="entity",
                    group_by_roles=[
                        "dimension",
                    ],
                    output_name=(
                        "distinct_entity_count"
                    ),
                )
            ),
        )


        return mapped(
            plan,
            contract,
        )


    if (
        kind ==
        "b2b_revenue_distribution"
    ):
        contract = AnalyticalContract(
            **base,
            family="aggregation",
            analytical_grain="entity",
            bindings=without_none(
                [
                    customer,
                    amount,
                ]
            ),
            aggregation=(
                AggregationSpec(
                    function="sum",
                    source_role="value",
                    group_by_roles=[
                        "entity",
                    ],
                    output_name=(
                        "aggregated_value"
                    ),
                )
            ),
        )


        return mapped(
            plan,
            contract,
            notes=[
                (
                    "Le segment demandé n'est volontairement "
                    "pas inventé. Si le legacy planner est "
                    "bloqué, le contrat générique reste bloqué."
                ),
            ],
        )


    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    if (
        kind in {
            "top_products",
            "flop_products",
        }
    ):
        product_dimension = (
            physical_binding(
                plan,
                role="dimension",
                concepts=(
                    "product_id",
                    "item_id",
                ),
                semantic_concept=(
                    "entity_identifier"
                ),
            )
        )


        contract = AnalyticalContract(
            **base,
            family="ranking",
            analytical_grain="entity",
            bindings=without_none(
                [
                    product_dimension,
                    amount,
                ]
            ),
            aggregation=(
                AggregationSpec(
                    function="sum",
                    source_role="value",
                    group_by_roles=[
                        "dimension",
                    ],
                    output_name=(
                        "ranking_measure"
                    ),
                )
            ),
            ranking=(
                RankingSpec(
                    order=(
                        "descending"
                        if kind ==
                        "top_products"
                        else
                        "ascending"
                    ),
                    limit=10,
                )
            ),
        )


        return mapped(
            plan,
            contract,
            notes=[
                (
                    "Le classement est désormais représenté "
                    "par une primitive générique dimension + "
                    "mesure + agrégation + ordre."
                ),
            ],
        )


    # --------------------------------------------------------
    # INEQUALITY
    # --------------------------------------------------------

    if (
        kind ==
        "lorenz_curve"
    ):
        contract = AnalyticalContract(
            **base,
            family="inequality",
            analytical_grain="entity",
            bindings=without_none(
                [
                    customer,
                    amount,
                ]
            ),
            aggregation=(
                AggregationSpec(
                    function="sum",
                    source_role="value",
                    group_by_roles=[
                        "entity",
                    ],
                    output_name=(
                        "entity_total"
                    ),
                )
            ),
        )


        return mapped(
            plan,
            contract,
        )


    # --------------------------------------------------------
    # CATEGORICAL ASSOCIATION
    # --------------------------------------------------------

    if (
        kind ==
        "gender_category_association"
    ):
        contract = AnalyticalContract(
            **base,
            family=(
                "categorical_association"
            ),
            analytical_grain="event",
            bindings=without_none(
                [
                    gender_x,
                    category_y,
                ]
            ),
        )


        return mapped(
            plan,
            contract,
        )


    # --------------------------------------------------------
    # AGE-BASED LEGACY REQUESTS
    #
    # The business-specific names exist only in this temporary
    # adapter. The resulting contracts use generic families,
    # generic roles and deterministic derived-variable specs.
    # --------------------------------------------------------

    if (
        kind ==
        "age_total_amount_association"
    ):
        age = derived_binding(
            role="x",
            column=(
                "age_at_first_purchase"
            ),
            semantic_concept=(
                "age_at_reference_event"
            ),
        )

        total = derived_binding(
            role="y",
            column=(
                "entity_total_value"
            ),
            semantic_concept=(
                "aggregated_measure"
            ),
        )


        derived = [
            *age_derived_variables(
                plan
            ),
            DerivedVariableSpec(
                output_column=(
                    "entity_total_value"
                ),
                operation="aggregate",
                source_columns=(
                    [
                        amount.column
                    ]
                    if amount is not None
                    else []
                ),
                group_by_columns=(
                    [
                        customer.column
                    ]
                    if customer is not None
                    else []
                ),
                parameters={
                    "function":
                        "sum",
                },
                rationale=(
                    "Construire la mesure totale au "
                    "grain entité avant l'association."
                ),
            ),
        ]


        contract = AnalyticalContract(
            **base,
            family=(
                "quantitative_association"
            ),
            analytical_grain="entity",
            bindings=[
                age,
                total,
            ],
            derived_variables=(
                derived
            ),
        )


        return mapped(
            plan,
            contract,
        )


    if (
        kind ==
        "age_frequency_association"
    ):
        session = find_match(
            plan,
            "session_id",
            "order_id",
        )

        age = derived_binding(
            role="x",
            column=(
                "age_at_first_purchase"
            ),
            semantic_concept=(
                "age_at_reference_event"
            ),
        )

        frequency = derived_binding(
            role="y",
            column=(
                "entity_event_frequency"
            ),
            semantic_concept=(
                "distinct_event_count"
            ),
        )


        contract = AnalyticalContract(
            **base,
            family=(
                "quantitative_association"
            ),
            analytical_grain="entity",
            bindings=[
                age,
                frequency,
            ],
            derived_variables=[
                *age_derived_variables(
                    plan
                ),
                DerivedVariableSpec(
                    output_column=(
                        "entity_event_frequency"
                    ),
                    operation="aggregate",
                    source_columns=(
                        [
                            session.column
                        ]
                        if session is not None
                        else []
                    ),
                    group_by_columns=(
                        [
                            customer.column
                        ]
                        if customer is not None
                        else []
                    ),
                    parameters={
                        "function":
                            "distinct_count",
                    },
                    rationale=(
                        "Construire une fréquence "
                        "d'événements au grain entité."
                    ),
                ),
            ],
        )


        return mapped(
            plan,
            contract,
        )


    if (
        kind ==
        "age_average_basket_association"
    ):
        session = find_match(
            plan,
            "session_id",
            "order_id",
        )

        age = derived_binding(
            role="x",
            column=(
                "age_at_first_purchase"
            ),
            semantic_concept=(
                "age_at_reference_event"
            ),
        )

        average = derived_binding(
            role="y",
            column=(
                "entity_average_event_value"
            ),
            semantic_concept=(
                "average_aggregated_measure"
            ),
        )


        contract = AnalyticalContract(
            **base,
            family=(
                "quantitative_association"
            ),
            analytical_grain="entity",
            bindings=[
                age,
                average,
            ],
            derived_variables=[
                *age_derived_variables(
                    plan
                ),
                DerivedVariableSpec(
                    output_column=(
                        "event_total_value"
                    ),
                    operation="aggregate",
                    source_columns=(
                        [
                            amount.column
                        ]
                        if amount is not None
                        else []
                    ),
                    group_by_columns=(
                        [
                            session.column
                        ]
                        if session is not None
                        else []
                    ),
                    parameters={
                        "function":
                            "sum",
                    },
                    rationale=(
                        "Calculer une valeur totale "
                        "par événement ou session."
                    ),
                ),
                DerivedVariableSpec(
                    output_column=(
                        "entity_average_event_value"
                    ),
                    operation="aggregate",
                    source_columns=[
                        "event_total_value",
                    ],
                    group_by_columns=(
                        [
                            customer.column
                        ]
                        if customer is not None
                        else []
                    ),
                    parameters={
                        "function":
                            "mean",
                    },
                    rationale=(
                        "Calculer la valeur moyenne des "
                        "événements au grain entité."
                    ),
                ),
            ],
        )


        return mapped(
            plan,
            contract,
        )


    if (
        kind ==
        "age_category_association"
    ):
        age_value = derived_binding(
            role="value",
            column=(
                "age_at_first_purchase"
            ),
            semantic_concept=(
                "age_at_reference_event"
            ),
        )


        contract = AnalyticalContract(
            **base,
            family=(
                "group_comparison"
            ),
            analytical_grain="event",
            bindings=without_none(
                [
                    category_group,
                    age_value,
                ]
            ),
            derived_variables=(
                age_derived_variables(
                    plan
                )
            ),
            reasons=[
                *base[
                    "reasons"
                ],
                (
                    "La relation quantitative × catégorie "
                    "est exprimée comme comparaison de groupes "
                    "dans le contrat générique."
                ),
            ],
        )


        return mapped(
            plan,
            contract,
        )


    # --------------------------------------------------------
    # UNKNOWN / FUTURE LEGACY KINDS
    # --------------------------------------------------------

    fallback_blockers = [
        *plan.blockers,
    ]


    if (
        len(
            fallback_blockers
        ) ==
        0
    ):
        fallback_blockers.append(
            (
                "Aucun mapping générique sûr n'est "
                "défini pour ce kind legacy."
            )
        )


    contract = AnalyticalContract(
        **{
            **base,
            "status":
                "ambiguous",
            "blockers":
                fallback_blockers,
        },
        family="unresolved",
        analytical_grain=None,
        bindings=[],
    )


    return LegacyContractMapping(
        request_id=(
            plan.request_id
        ),
        legacy_kind=(
            str(
                plan.kind
            )
        ),
        mapping_status=(
            "ambiguous"
        ),
        contract=(
            contract
        ),
        adapter_notes=[
            (
                "Le kind legacy n'est pas encore "
                "convertible sans hypothèse."
            ),
        ],
    )


def mapped(
    plan: RequestedAnalysisPlan,
    contract: AnalyticalContract,
    *,
    notes: (
        list[
            str
        ]
        | None
    ) = None,
) -> LegacyContractMapping:
    if (
        contract.status ==
        "blocked"
    ):
        status: LegacyMappingStatus = (
            "blocked"
        )

    elif (
        contract.status ==
        "ambiguous"
    ):
        status = (
            "ambiguous"
        )

    else:
        status = (
            "mapped"
        )


    return LegacyContractMapping(
        request_id=(
            plan.request_id
        ),
        legacy_kind=(
            str(
                plan.kind
            )
        ),
        mapping_status=(
            status
        ),
        contract=(
            contract
        ),
        adapter_notes=(
            notes
            or []
        ),
    )


# ============================================================
# REPORT ADAPTER
# ============================================================

def build_legacy_contract_migration_report(
    plan_report: RequestedAnalysisPlanReport,
) -> LegacyContractMigrationReport:
    mappings = [
        adapt_requested_plan(
            plan
        )
        for plan
        in plan_report.requests
    ]


    return LegacyContractMigrationReport(
        request_count=(
            len(
                mappings
            )
        ),
        mapped_count=sum(
            1
            for mapping
            in mappings
            if (
                mapping.mapping_status ==
                "mapped"
            )
        ),
        blocked_count=sum(
            1
            for mapping
            in mappings
            if (
                mapping.mapping_status ==
                "blocked"
            )
        ),
        ambiguous_count=sum(
            1
            for mapping
            in mappings
            if (
                mapping.mapping_status ==
                "ambiguous"
            )
        ),
        mappings=(
            mappings
        ),
    )
