from typing import Any

from app.ai.provider import (
    DEFAULT_MODEL,
    generate_dataset_explanation,
)
from app.ai.schemas import (
    DatasetAIExplanation,
)
from app.statistics.schemas import (
    CorrelationAnalysis,
    CorrelationResult,
)


MAX_GROUNDING_ATTEMPTS = 2


class GroundingValidationError(
    RuntimeError
):
    """
    Raised when an AI response is not grounded
    in deterministic DataLens evidence.
    """

    pass


def build_evidence_id(
    prefix: str,
    index: int,
) -> str:
    return (
        f"{prefix}:"
        f"{index:04d}"
    )


def canonicalize_claim_value(
    value: Any,
) -> str:
    """
    Convert deterministic Python values to
    canonical claim strings.

    Float identity is preserved:
    -1.0 remains "-1.0".
    """

    if value is None:
        return "null"

    if isinstance(
        value,
        bool,
    ):
        return (
            "true"
            if value
            else "false"
        )

    if isinstance(
        value,
        float,
    ):
        return repr(
            value
        )

    return str(
        value
    )


# ============================================================
# CLEANING CONTEXT
# ============================================================

def compact_cleaning_operation(
    operation: dict[str, Any],
) -> dict[str, Any]:
    allowed_fields = (
        "operation",
        "column",
        "affected_rows",
        "affected_values",
        "affected_columns",
        "from_dtype",
        "to_dtype",
        "automatic",
        "reversible",
        "reason",
    )

    return {
        field:
            operation[
                field
            ]
        for field
        in allowed_fields
        if field in operation
    }


def compact_review_item(
    item: dict[str, Any],
) -> dict[str, Any]:
    allowed_fields = (
        "type",
        "column",
        "status",
        "automatic_change",
        "missing_rate",
        "reason",
        "suggested_action",
    )

    return {
        field:
            item[
                field
            ]
        for field
        in allowed_fields
        if field in item
    }


def build_cleaning_context(
    cleaning_report: dict[str, Any],
) -> dict[str, Any]:
    filename = (
        cleaning_report.get(
            "filename"
        )
    )

    if not filename:
        raise ValueError(
            "Cleaning report has no filename."
        )

    evidence = []

    operations = (
        cleaning_report.get(
            "operations",
            [],
        )
    )

    for (
        index,
        operation,
    ) in enumerate(
        operations,
        start=1,
    ):
        evidence.append(
            {
                "evidence_id":
                    build_evidence_id(
                        prefix="cleaning",
                        index=index,
                    ),

                "source_type":
                    "cleaning_operation",

                "data":
                    compact_cleaning_operation(
                        operation
                    ),
            }
        )

    review_items = (
        cleaning_report.get(
            "review_required",
            [],
        )
    )

    for (
        index,
        item,
    ) in enumerate(
        review_items,
        start=1,
    ):
        evidence.append(
            {
                "evidence_id":
                    build_evidence_id(
                        prefix="review",
                        index=index,
                    ),

                "source_type":
                    "review_item",

                "data":
                    compact_review_item(
                        item
                    ),
            }
        )

    return {
        "dataset":
            str(
                filename
            ),

        "evidence":
            evidence,
    }


# ============================================================
# STATISTICAL CONTEXT
# ============================================================

def build_correlation_evidence_data(
    analysis: CorrelationAnalysis,
    result: CorrelationResult,
) -> dict[str, Any]:
    """
    Convert one deterministic correlation
    result into evidence consumable by the LLM.
    """

    return {
        "x_column":
            analysis.x_column,

        "y_column":
            analysis.y_column,

        "n_total":
            analysis.n_total,

        "n_valid":
            analysis.n_valid,

        "n_excluded":
            analysis.n_excluded,

        "test":
            result.test,

        "relationship_type":
            result.relationship_type,

        "coefficient_name":
            result.coefficient_name,

        "coefficient":
            result.coefficient,

        "p_value":
            result.p_value,

        "alternative":
            result.alternative,

        "n":
            result.n,

        "alpha":
            result.alpha,

        "statistically_significant":
            result.statistically_significant,

        "warnings":
            list(
                analysis.warnings
            ),
    }


def build_correlation_context(
    dataset: str,
    analysis: CorrelationAnalysis,
) -> dict[str, Any]:
    """
    Convert a deterministic CorrelationAnalysis
    into two statistical evidence items.

    statistic:0001 -> Pearson
    statistic:0002 -> Spearman
    """

    if not dataset:
        raise ValueError(
            "Dataset name cannot be empty."
        )

    evidence = [
        {
            "evidence_id":
                build_evidence_id(
                    prefix="statistic",
                    index=1,
                ),

            "source_type":
                "statistical_result",

            "data":
                build_correlation_evidence_data(
                    analysis=
                        analysis,

                    result=
                        analysis.pearson,
                ),
        },
        {
            "evidence_id":
                build_evidence_id(
                    prefix="statistic",
                    index=2,
                ),

            "source_type":
                "statistical_result",

            "data":
                build_correlation_evidence_data(
                    analysis=
                        analysis,

                    result=
                        analysis.spearman,
                ),
        },
    ]

    return {
        "dataset":
            dataset,

        "evidence":
            evidence,
    }


# ============================================================
# EVIDENCE REGISTRY
# ============================================================

def build_allowed_evidence(
    context: dict[str, Any],
) -> dict[str, set[str]]:
    allowed: dict[
        str,
        set[str],
    ] = {
        "cleaning_operation":
            set(),

        "review_item":
            set(),

        "profile_metric":
            set(),

        "statistical_result":
            set(),

        "relationship_result":
            set(),
    }

    for evidence in context.get(
        "evidence",
        [],
    ):
        source_type = (
            evidence.get(
                "source_type"
            )
        )

        evidence_id = (
            evidence.get(
                "evidence_id"
            )
        )

        if (
            source_type
            and evidence_id
            and source_type
            in allowed
        ):
            allowed[
                source_type
            ].add(
                str(
                    evidence_id
                )
            )

    return allowed


def build_evidence_index(
    context: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    index = {}

    for evidence in context.get(
        "evidence",
        [],
    ):
        evidence_id = (
            evidence.get(
                "evidence_id"
            )
        )

        if evidence_id:
            index[
                str(
                    evidence_id
                )
            ] = evidence

    return index


def build_required_finding_evidence_ids(
    context: dict[str, Any],
) -> set[str]:
    """
    Evidence types that must explicitly
    appear in findings.
    """

    required_source_types = {
        "cleaning_operation",
        "statistical_result",
    }

    return {
        str(
            evidence.get(
                "evidence_id"
            )
        )

        for evidence
        in context.get(
            "evidence",
            [],
        )

        if (
            evidence.get(
                "evidence_id"
            )
            and evidence.get(
                "source_type"
            )
            in required_source_types
        )
    }


# ============================================================
# GROUNDING VALIDATION
# ============================================================

def validate_grounding(
    explanation: DatasetAIExplanation,
    context: dict[str, Any],
) -> None:
    errors = []

    expected_dataset = (
        context.get(
            "dataset"
        )
    )

    if (
        expected_dataset
        and explanation.dataset
        != expected_dataset
    ):
        errors.append(
            (
                "Dataset mismatch: "
                f"expected '{expected_dataset}', "
                f"received "
                f"'{explanation.dataset}'."
            )
        )

    allowed = (
        build_allowed_evidence(
            context
        )
    )

    evidence_index = (
        build_evidence_index(
            context
        )
    )

    required_evidence_ids = (
        build_required_finding_evidence_ids(
            context
        )
    )

    if (
        required_evidence_ids
        and not explanation.findings
    ):
        errors.append(
            (
                "Required evidence exists, "
                "but the LLM returned no findings."
            )
        )

    used_finding_references = set()

    for (
        finding_index,
        finding,
    ) in enumerate(
        explanation.findings
    ):
        if not finding.evidence:
            errors.append(
                (
                    f"Finding {finding_index} "
                    "contains no evidence."
                )
            )

            continue

        cited_references = set()

        for evidence_reference in (
            finding.evidence
        ):
            reference = (
                evidence_reference
                .reference
            )

            source_type = (
                evidence_reference
                .source_type
            )

            allowed_references = (
                allowed.get(
                    source_type,
                    set(),
                )
            )

            if (
                reference
                not in allowed_references
            ):
                errors.append(
                    (
                        "Unsupported evidence "
                        f"in finding "
                        f"{finding_index}: "
                        f"{source_type}:"
                        f"{reference}"
                    )
                )

                continue

            actual_evidence = (
                evidence_index.get(
                    reference
                )
            )

            if actual_evidence is None:
                errors.append(
                    (
                        "Unknown evidence ID: "
                        f"{reference}"
                    )
                )

                continue

            actual_source_type = (
                actual_evidence.get(
                    "source_type"
                )
            )

            if (
                actual_source_type
                != source_type
            ):
                errors.append(
                    (
                        "Evidence source type "
                        f"mismatch for "
                        f"{reference}: "
                        f"expected "
                        f"{actual_source_type}, "
                        f"received "
                        f"{source_type}."
                    )
                )

                continue

            cited_references.add(
                reference
            )

            used_finding_references.add(
                reference
            )

        if not finding.claims:
            errors.append(
                (
                    f"Finding {finding_index} "
                    "contains no claims."
                )
            )

            continue

        for (
            claim_index,
            claim,
        ) in enumerate(
            finding.claims
        ):
            if (
                claim.reference
                not in cited_references
            ):
                errors.append(
                    (
                        f"Claim {claim_index} "
                        f"in finding "
                        f"{finding_index} "
                        "references evidence "
                        "not cited by the finding: "
                        f"{claim.reference}"
                    )
                )

                continue

            evidence = (
                evidence_index.get(
                    claim.reference
                )
            )

            if evidence is None:
                errors.append(
                    (
                        "Unknown evidence "
                        f"in claim: "
                        f"{claim.reference}"
                    )
                )

                continue

            data = (
                evidence.get(
                    "data",
                    {}
                )
            )

            if (
                claim.field
                not in data
            ):
                errors.append(
                    (
                        "Claim field "
                        f"'{claim.field}' "
                        "does not exist in "
                        f"{claim.reference}."
                    )
                )

                continue

            actual_value = (
                canonicalize_claim_value(
                    data[
                        claim.field
                    ]
                )
            )

            claimed_value = (
                claim.value
                .strip()
            )

            if (
                actual_value
                != claimed_value
            ):
                errors.append(
                    (
                        "Claim value mismatch "
                        f"in finding "
                        f"{finding_index}: "
                        f"{claim.reference} / "
                        f"{claim.field}; "
                        f"expected "
                        f"{actual_value!r}, "
                        f"received "
                        f"{claimed_value!r}."
                    )
                )

    missing_evidence = (
        required_evidence_ids
        - used_finding_references
    )

    if missing_evidence:
        errors.append(
            (
                "Required evidence was not covered "
                "by findings: "
                + ", ".join(
                    sorted(
                        missing_evidence
                    )
                )
            )
        )

    for (
        warning_index,
        warning,
    ) in enumerate(
        explanation.warnings
    ):
        for evidence_reference in (
            warning.evidence
        ):
            reference = (
                evidence_reference
                .reference
            )

            source_type = (
                evidence_reference
                .source_type
            )

            allowed_references = (
                allowed.get(
                    source_type,
                    set(),
                )
            )

            if (
                reference
                not in allowed_references
            ):
                errors.append(
                    (
                        "Unsupported evidence "
                        f"in warning "
                        f"{warning_index}: "
                        f"{source_type}:"
                        f"{reference}"
                    )
                )

                continue

            actual_evidence = (
                evidence_index.get(
                    reference
                )
            )

            if actual_evidence is None:
                errors.append(
                    (
                        "Unknown evidence ID "
                        f"in warning: "
                        f"{reference}"
                    )
                )

                continue

            if (
                actual_evidence.get(
                    "source_type"
                )
                != source_type
            ):
                errors.append(
                    (
                        "Evidence source type "
                        "mismatch in warning "
                        f"{warning_index}: "
                        f"{reference}"
                    )
                )

    if errors:
        raise GroundingValidationError(
            "\n".join(
                errors
            )
        )


# ============================================================
# GENERIC AI PIPELINE
# ============================================================

def explain_dataset(
    context: dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> DatasetAIExplanation:
    last_error = None

    for attempt in range(
        MAX_GROUNDING_ATTEMPTS
    ):
        explanation = (
            generate_dataset_explanation(
                context=
                    context,

                model=
                    model,

                strict_retry=(
                    attempt > 0
                ),
            )
        )

        try:
            validate_grounding(
                explanation=
                    explanation,

                context=
                    context,
            )

            return explanation

        except GroundingValidationError as error:
            last_error = error

    raise GroundingValidationError(
        (
            "The local LLM failed DataLens "
            "grounding validation after "
            f"{MAX_GROUNDING_ATTEMPTS} attempts.\n"
            f"{last_error}"
        )
    )


# ============================================================
# CLEANING AI
# ============================================================

def explain_cleaning_report(
    cleaning_report: dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> DatasetAIExplanation:
    context = (
        build_cleaning_context(
            cleaning_report
        )
    )

    return explain_dataset(
        context=context,
        model=model,
    )


# ============================================================
# STATISTICAL AI
# ============================================================

def explain_correlation_analysis(
    dataset: str,
    analysis: CorrelationAnalysis,
    model: str = DEFAULT_MODEL,
) -> DatasetAIExplanation:
    """
    Explain deterministic Pearson and Spearman
    results using the guarded AI pipeline.
    """

    context = (
        build_correlation_context(
            dataset=dataset,
            analysis=analysis,
        )
    )

    return explain_dataset(
        context=context,
        model=model,
    )