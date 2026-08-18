from __future__ import annotations

import json
import re

from app.ai.provider import (
    DEFAULT_MODEL,
    client,
)

from app.semantics.family_schemas import (
    DatasetQuantityFamilyReport,
    QuantityFamilyAssignment,
    QuantityFamilyClusteringDraft,
    QuantityFamilyRelationDecision,
)

from app.semantics.quantity import (
    QUANTITY_UNIT_TOKENS,
    is_numeric_quantity_dimension,
    normalize_quantity_text,
)

from app.semantics.schemas import (
    ColumnSemanticProfile,
    DatasetSemanticProfile,
)


# ============================================================
# VERSIONS
# ============================================================

QUANTITY_FAMILY_RULE_VERSION = (
    "quantity_family_clustering_v0.2"
)


QUANTITY_FAMILY_RECONCILIATION_RULE_VERSION = (
    "hybrid_quantity_family_reconciler_v0.1"
)


MAX_CLUSTERING_ATTEMPTS = 2


UNKNOWN = (
    "unknown"
)


# ============================================================
# NUMERIC MEASURES
# ============================================================

NUMERIC_MEASURE_KINDS = {
    "count",
    "rate",
    "percentage",
    "index",
    "currency",
    "duration",
}


# ============================================================
# PROMPT
# ============================================================

QUANTITY_FAMILY_SYSTEM_PROMPT = """
You are DataLens' dataset-level semantic quantity-family
clustering component.

You receive normalized semantic profiles for quantitative
columns from ONE dataset.

Assign every supplied column to exactly one quantity_family.

DEFINITION

A quantity_family represents WHAT underlying measurable
business quantity a column represents.

State, stage, version or observation must remain conceptually
separate from the underlying quantity.

For example, words that represent an intended, observed,
requested, granted, expected, completed, previous or current
state should not by themselves create different quantity
families when the underlying measurable thing is the same.

IMPORTANT NEGATIVE RULES

Do NOT place two columns in the same quantity family merely
because:

- they have the same numeric type;
- they have the same mathematical dimension;
- they belong to the same business domain;
- they belong to the same business process;
- they may be correlated;
- one may influence the other;
- comparing them may be analytically useful.

Two counts can count different entities.

Two monetary values can represent different economic
quantities.

Two percentages can measure different phenomena.

STATE-ABSTRACTED SIGNATURE

Each column includes a deterministic
state_abstracted_signature.

It is evidence about the underlying quantity after removing
an already detected state and explicit unit notation.

Use it as evidence, but do not treat it as infallible.

EXISTING SEMANTIC FIELDS

concept, semantic_group and state are hypotheses produced by
earlier DataLens components.

Use them as evidence.

Do not blindly copy them.

OUTPUT CONTRACT

- Return every supplied column exactly once.
- Copy each original column name exactly.
- Never duplicate a column.
- Never omit a column.
- Assign exactly one quantity_family per column.
- Use concise lowercase snake_case family labels.
- Do not return broad domain names as quantity families.
- Do not determine unit compatibility.
- Do not determine subtraction safety.
- Do not determine derived-gap compatibility.
- Do not calculate metrics.

Return only the structured response.
""".strip()


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_family_label(
    value: str,
) -> str:
    normalized = (
        normalize_quantity_text(
            value
        )
    )


    return (
        normalized
        or
        UNKNOWN
    )


def remove_parenthetical_content(
    value: str,
) -> str:
    return re.sub(
        r"\([^)]*\)",
        " ",
        str(
            value
        ),
    )


# ============================================================
# STATE-ABSTRACTED SIGNATURE
# ============================================================

def build_state_abstracted_signature(
    *,
    column: str,
    variant: str,
) -> str:
    without_parentheses = (
        remove_parenthetical_content(
            column
        )
    )


    normalized = (
        normalize_quantity_text(
            without_parentheses
        )
    )


    tokens = [
        token

        for token
        in normalized.split(
            "_"
        )

        if token
    ]


    normalized_variant = (
        normalize_family_label(
            variant
        )
    )


    if (
        normalized_variant
        !=
        UNKNOWN
    ):
        variant_tokens = set(
            normalized_variant.split(
                "_"
            )
        )


        tokens = [
            token

            for token
            in tokens

            if (
                token
                not in variant_tokens
            )
        ]


    tokens = [
        token

        for token
        in tokens

        if (
            token
            not in QUANTITY_UNIT_TOKENS
        )
    ]


    signature = "_".join(
        tokens
    )


    return (
        signature
        or
        normalize_family_label(
            column
        )
    )


# ============================================================
# ELIGIBILITY — S4.1
#
# S4 v0.1 required S3 to have already recognized either:
#
# - a known quantitative measure kind, or
# - a known quantitative dimension.
#
# This caused cascading false negatives:
#
# a structurally quantitative measure could be hidden from S4
# simply because S3 had uncertain semantic metadata.
#
# S4.1 adds an independent structural route:
#
#     entity_role == "measure"
#     AND
#     data_type == "quantitative"
#
# Structural numerical evidence is enough to let S4 inspect a
# column. It is NOT enough to authorize arithmetic.
# ============================================================

def profile_is_quantity_family_eligible(
    profile: ColumnSemanticProfile,
) -> bool:
    if (
        profile.entity_role
        !=
        "measure"
    ):
        return False


    if (
        profile.data_type
        ==
        "quantitative"
    ):
        return True


    if (
        profile.measure_kind
        in NUMERIC_MEASURE_KINDS
    ):
        return True


    return (
        is_numeric_quantity_dimension(
            profile.quantity_dimension
        )
    )


def eligible_profiles(
    dataset_profile: DatasetSemanticProfile,
) -> list[
    ColumnSemanticProfile
]:
    return [
        profile

        for profile
        in dataset_profile.columns

        if (
            profile_is_quantity_family_eligible(
                profile
            )
        )
    ]


# ============================================================
# MODEL PAYLOAD
# ============================================================

def build_quantity_family_payload(
    profile: ColumnSemanticProfile,
) -> dict[
    str,
    str,
]:
    return {
        "column":
            profile.column,

        "concept":
            profile.concept,

        "semantic_group":
            profile.semantic_group,

        "state":
            profile.variant,

        "state_abstracted_signature":
            build_state_abstracted_signature(
                column=
                    profile.column,

                variant=
                    profile.variant,
            ),
    }


# ============================================================
# STRUCTURAL VALIDATION
# ============================================================

def validate_clustering_coverage(
    *,
    draft: QuantityFamilyClusteringDraft,
    profiles: list[
        ColumnSemanticProfile
    ],
) -> None:
    expected_columns = {
        profile.column

        for profile
        in profiles
    }


    returned_columns = [
        assignment.column

        for assignment
        in draft.assignments
    ]


    returned_set = set(
        returned_columns
    )


    if (
        len(
            returned_columns
        )
        !=
        len(
            returned_set
        )
    ):
        raise ValueError(
            "Quantity-family clustering returned "
            "duplicate columns."
        )


    if (
        returned_set
        !=
        expected_columns
    ):
        missing = (
            expected_columns
            -
            returned_set
        )


        extra = (
            returned_set
            -
            expected_columns
        )


        raise ValueError(
            "Quantity-family clustering column coverage "
            "mismatch. "
            f"Missing={sorted(missing)}. "
            f"Extra={sorted(extra)}."
        )


# ============================================================
# LLM CLUSTERING
# ============================================================

def call_quantity_family_model(
    *,
    dataset_profile: DatasetSemanticProfile,
    profiles: list[
        ColumnSemanticProfile
    ],
    model: str,
) -> QuantityFamilyClusteringDraft:
    payload = {
        "dataset_id":
            dataset_profile.dataset_id,

        "filename":
            dataset_profile.filename,

        "columns": [
            build_quantity_family_payload(
                profile
            )

            for profile
            in profiles
        ],
    }


    last_error: Exception | None = None


    for attempt in range(
        MAX_CLUSTERING_ATTEMPTS
    ):
        try:
            retry_instruction = (
                ""
                if (
                    attempt
                    ==
                    0
                )
                else
                (
                    "\n\nSTRICT RETRY:\n"
                    "Return every supplied column exactly "
                    "once and obey the output schema."
                )
            )


            response = (
                client.chat(
                    model=
                        model,

                    messages=[
                        {
                            "role":
                                "system",

                            "content": (
                                QUANTITY_FAMILY_SYSTEM_PROMPT
                                +
                                retry_instruction
                            ),
                        },
                        {
                            "role":
                                "user",

                            "content": (
                                "Cluster this complete "
                                "quantitative semantic "
                                "dataset:\n\n"
                                +
                                json.dumps(
                                    payload,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                            ),
                        },
                    ],

                    format=(
                        QuantityFamilyClusteringDraft
                        .model_json_schema()
                    ),

                    options={
                        "temperature":
                            0.0,
                    },
                )
            )


            draft = (
                QuantityFamilyClusteringDraft
                .model_validate_json(
                    response.message.content
                )
            )


            validate_clustering_coverage(
                draft=
                    draft,

                profiles=
                    profiles,
            )


            return draft


        except Exception as error:
            last_error = error


    raise RuntimeError(
        "Quantity-family clustering failed after "
        f"{MAX_CLUSTERING_ATTEMPTS} attempts. "
        f"Last error: {last_error}"
    )


# ============================================================
# FALLBACK
# ============================================================

def fallback_assignment(
    *,
    dataset_profile: DatasetSemanticProfile,
    profile: ColumnSemanticProfile,
    reason: str,
) -> QuantityFamilyAssignment:
    signature = (
        build_state_abstracted_signature(
            column=
                profile.column,

            variant=
                profile.variant,
        )
    )


    return QuantityFamilyAssignment(
        dataset_id=
            dataset_profile.dataset_id,

        filename=
            dataset_profile.filename,

        column=
            profile.column,

        quantity_family=
            signature,

        state=
            profile.variant,

        state_abstracted_signature=
            signature,

        quantity_dimension=
            profile.quantity_dimension,

        quantity_unit=
            profile.quantity_unit,

        confidence=
            "low",

        source=
            "deterministic_fallback",

        reason=
            reason,

        quantity_family_rule_version=
            QUANTITY_FAMILY_RULE_VERSION,
    )


# ============================================================
# DATASET FAMILY REPORT
# ============================================================

def build_dataset_quantity_family_report(
    *,
    dataset_profile: DatasetSemanticProfile,
    model: str = DEFAULT_MODEL,
) -> DatasetQuantityFamilyReport:
    profiles = (
        eligible_profiles(
            dataset_profile
        )
    )


    if not profiles:
        return DatasetQuantityFamilyReport(
            dataset_id=
                dataset_profile.dataset_id,

            filename=
                dataset_profile.filename,

            eligible_column_count=
                0,

            assignment_count=
                0,

            family_count=
                0,

            llm_assignment_count=
                0,

            fallback_assignment_count=
                0,

            clustering_succeeded=
                True,

            model=
                model,

            assignments=
                [],

            quantity_family_rule_version=
                QUANTITY_FAMILY_RULE_VERSION,
        )


    if (
        len(
            profiles
        )
        ==
        1
    ):
        profile = (
            profiles[
                0
            ]
        )


        assignment = (
            fallback_assignment(
                dataset_profile=
                    dataset_profile,

                profile=
                    profile,

                reason=(
                    "Only one eligible quantitative column "
                    "exists in this dataset; no semantic "
                    "clustering call is required."
                ),
            )
        )


        return DatasetQuantityFamilyReport(
            dataset_id=
                dataset_profile.dataset_id,

            filename=
                dataset_profile.filename,

            eligible_column_count=
                1,

            assignment_count=
                1,

            family_count=
                1,

            llm_assignment_count=
                0,

            fallback_assignment_count=
                1,

            clustering_succeeded=
                True,

            model=
                model,

            assignments=[
                assignment
            ],

            quantity_family_rule_version=
                QUANTITY_FAMILY_RULE_VERSION,
        )


    try:
        draft = (
            call_quantity_family_model(
                dataset_profile=
                    dataset_profile,

                profiles=
                    profiles,

                model=
                    model,
            )
        )


        draft_index = {
            assignment.column:
                assignment

            for assignment
            in draft.assignments
        }


        assignments = []


        for profile in profiles:
            item = (
                draft_index[
                    profile.column
                ]
            )


            signature = (
                build_state_abstracted_signature(
                    column=
                        profile.column,

                    variant=
                        profile.variant,
                )
            )


            assignments.append(
                QuantityFamilyAssignment(
                    dataset_id=
                        dataset_profile.dataset_id,

                    filename=
                        dataset_profile.filename,

                    column=
                        profile.column,

                    quantity_family=
                        normalize_family_label(
                            item.quantity_family
                        ),

                    state=
                        profile.variant,

                    state_abstracted_signature=
                        signature,

                    quantity_dimension=
                        profile.quantity_dimension,

                    quantity_unit=
                        profile.quantity_unit,

                    confidence=
                        item.confidence,

                    source=
                        "llm",

                    reason=
                        item.reason,

                    quantity_family_rule_version=
                        QUANTITY_FAMILY_RULE_VERSION,
                )
            )


        family_count = len(
            {
                assignment.quantity_family

                for assignment
                in assignments
            }
        )


        return DatasetQuantityFamilyReport(
            dataset_id=
                dataset_profile.dataset_id,

            filename=
                dataset_profile.filename,

            eligible_column_count=
                len(
                    profiles
                ),

            assignment_count=
                len(
                    assignments
                ),

            family_count=
                family_count,

            llm_assignment_count=
                len(
                    assignments
                ),

            fallback_assignment_count=
                0,

            clustering_succeeded=
                True,

            model=
                model,

            assignments=
                assignments,

            quantity_family_rule_version=
                QUANTITY_FAMILY_RULE_VERSION,
        )


    except Exception as error:
        assignments = [
            fallback_assignment(
                dataset_profile=
                    dataset_profile,

                profile=
                    profile,

                reason=(
                    "LLM quantity-family clustering was "
                    "unavailable. Conservative deterministic "
                    f"fallback used. Error: {error}"
                ),
            )

            for profile
            in profiles
        ]


        return DatasetQuantityFamilyReport(
            dataset_id=
                dataset_profile.dataset_id,

            filename=
                dataset_profile.filename,

            eligible_column_count=
                len(
                    profiles
                ),

            assignment_count=
                len(
                    assignments
                ),

            family_count=
                len(
                    {
                        assignment.quantity_family

                        for assignment
                        in assignments
                    }
                ),

            llm_assignment_count=
                0,

            fallback_assignment_count=
                len(
                    assignments
                ),

            clustering_succeeded=
                False,

            model=
                model,

            assignments=
                assignments,

            quantity_family_rule_version=
                QUANTITY_FAMILY_RULE_VERSION,
        )


# ============================================================
# MULTI-DATASET FAMILY REPORTS
# ============================================================

def build_quantity_family_reports(
    *,
    profiles: list[
        DatasetSemanticProfile
    ],
    model: str = DEFAULT_MODEL,
) -> list[
    DatasetQuantityFamilyReport
]:
    return [
        build_dataset_quantity_family_report(
            dataset_profile=
                dataset_profile,

            model=
                model,
        )

        for dataset_profile
        in profiles
    ]


# ============================================================
# ASSIGNMENT LOOKUP
# ============================================================

def quantity_family_assignment_by_column(
    *,
    report: DatasetQuantityFamilyReport,
    column: str,
) -> QuantityFamilyAssignment | None:
    for assignment in (
        report.assignments
    ):
        if (
            assignment.column
            ==
            column
        ):
            return assignment


    return None


# ============================================================
# DIMENSION CONFLICT
# ============================================================

def profiles_have_dimension_conflict(
    *,
    left: ColumnSemanticProfile,
    right: ColumnSemanticProfile,
) -> bool:
    left_dimension = (
        normalize_family_label(
            left.quantity_dimension
        )
    )


    right_dimension = (
        normalize_family_label(
            right.quantity_dimension
        )
    )


    if (
        left_dimension
        ==
        UNKNOWN
        or
        right_dimension
        ==
        UNKNOWN
    ):
        return False


    return (
        left_dimension
        !=
        right_dimension
    )


# ============================================================
# DISTINCT KNOWN STATES
# ============================================================

def profiles_have_distinct_known_states(
    *,
    left: ColumnSemanticProfile,
    right: ColumnSemanticProfile,
) -> bool:
    left_state = (
        normalize_family_label(
            left.variant
        )
    )


    right_state = (
        normalize_family_label(
            right.variant
        )
    )


    return (
        left_state
        !=
        UNKNOWN

        and

        right_state
        !=
        UNKNOWN

        and

        left_state
        !=
        right_state
    )


# ============================================================
# HYBRID PAIR RECONCILIATION
# ============================================================

def reconcile_quantity_family_pair(
    *,
    left: ColumnSemanticProfile,
    right: ColumnSemanticProfile,
    report: DatasetQuantityFamilyReport,
) -> QuantityFamilyRelationDecision:
    left_assignment = (
        quantity_family_assignment_by_column(
            report=
                report,

            column=
                left.column,
        )
    )


    right_assignment = (
        quantity_family_assignment_by_column(
            report=
                report,

            column=
                right.column,
        )
    )


    left_signature = (
        build_state_abstracted_signature(
            column=
                left.column,

            variant=
                left.variant,
        )
    )


    right_signature = (
        build_state_abstracted_signature(
            column=
                right.column,

            variant=
                right.variant,
        )
    )


    signature_same = (
        left_signature
        ==
        right_signature
    )


    distinct_states = (
        profiles_have_distinct_known_states(
            left=
                left,

            right=
                right,
        )
    )


    dimension_conflict = (
        profiles_have_dimension_conflict(
            left=
                left,

            right=
                right,
        )
    )


    left_family = (
        left_assignment.quantity_family
        if (
            left_assignment
            is not None
        )
        else
        UNKNOWN
    )


    right_family = (
        right_assignment.quantity_family
        if (
            right_assignment
            is not None
        )
        else
        UNKNOWN
    )


    llm_same_family = (
        left_assignment
        is not None

        and

        right_assignment
        is not None

        and

        left_assignment.source
        ==
        "llm"

        and

        right_assignment.source
        ==
        "llm"

        and

        left_family
        !=
        UNKNOWN

        and

        left_family
        ==
        right_family
    )


    reasons: list[
        str
    ] = []


    # --------------------------------------------------------
    # SAFETY FIREWALL
    # --------------------------------------------------------

    if dimension_conflict:
        reasons.append(
            (
                "Les dimensions quantitatives connues sont "
                "différentes ; la relation de même quantité "
                "est rejetée déterministiquement."
            )
        )


        return QuantityFamilyRelationDecision(
            left_dataset_id=
                left.dataset_id,

            left_column=
                left.column,

            right_dataset_id=
                right.dataset_id,

            right_column=
                right.column,

            same_quantity_family=
                False,

            source=
                "dimension_veto",

            left_family=
                left_family,

            right_family=
                right_family,

            left_signature=
                left_signature,

            right_signature=
                right_signature,

            left_state=
                left.variant,

            right_state=
                right.variant,

            left_quantity_dimension=
                left.quantity_dimension,

            right_quantity_dimension=
                right.quantity_dimension,

            llm_same_family=
                llm_same_family,

            signature_same=
                signature_same,

            distinct_known_states=
                distinct_states,

            dimension_conflict=
                True,

            reasons=
                reasons,

            reconciliation_rule_version=
                QUANTITY_FAMILY_RECONCILIATION_RULE_VERSION,
        )


    deterministic_bridge = (
        signature_same
        and
        distinct_states
    )


    if (
        llm_same_family
        and
        deterministic_bridge
    ):
        reasons.append(
            (
                "Le clustering sémantique et la signature "
                "abstraite de l'état convergent vers la même "
                "famille quantitative."
            )
        )


        source = (
            "llm_plus_signature"
        )


        same_quantity_family = (
            True
        )


    elif deterministic_bridge:
        reasons.append(
            (
                "Les colonnes ont la même signature après "
                "abstraction de deux états distincts connus."
            )
        )


        source = (
            "state_abstracted_signature"
        )


        same_quantity_family = (
            True
        )


    elif llm_same_family:
        reasons.append(
            (
                "Le clustering sémantique place les deux "
                "colonnes dans la même famille quantitative."
            )
        )


        source = (
            "llm_cluster"
        )


        same_quantity_family = (
            True
        )


    else:
        reasons.append(
            (
                "Aucune preuve positive suffisante de même "
                "famille quantitative n'a été établie."
            )
        )


        source = (
            "no_positive_evidence"
        )


        same_quantity_family = (
            False
        )


    if (
        left_assignment
        is None
        or
        right_assignment
        is None
    ):
        source = (
            "unavailable"
        )


        same_quantity_family = (
            False
        )


        reasons = [
            (
                "Au moins une colonne ne possède pas "
                "d'affectation de famille quantitative."
            )
        ]


    return QuantityFamilyRelationDecision(
        left_dataset_id=
            left.dataset_id,

        left_column=
            left.column,

        right_dataset_id=
            right.dataset_id,

        right_column=
            right.column,

        same_quantity_family=
            same_quantity_family,

        source=
            source,

        left_family=
            left_family,

        right_family=
            right_family,

        left_signature=
            left_signature,

        right_signature=
            right_signature,

        left_state=
            left.variant,

        right_state=
            right.variant,

        left_quantity_dimension=
            left.quantity_dimension,

        right_quantity_dimension=
            right.quantity_dimension,

        llm_same_family=
            llm_same_family,

        signature_same=
            signature_same,

        distinct_known_states=
            distinct_states,

        dimension_conflict=
            dimension_conflict,

        reasons=
            reasons,

        reconciliation_rule_version=
            QUANTITY_FAMILY_RECONCILIATION_RULE_VERSION,
    )
