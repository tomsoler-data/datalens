from __future__ import annotations


from typing import (
    List,
)


from pydantic import (
    BaseModel,
    Field,
)


from app.preparation.analysis_output_selection import (
    AnalysisOutputSelectionReport,
    evaluate_analysis_output_selection,
)


from app.preparation.final_validation import (
    evaluate_final_preparation_validation as evaluate_final_preparation_validation_v0_1,
)


from app.preparation.preparation_session import (
    PreparationSessionView,
)


# ============================================================
# VERSION
# ============================================================


FINAL_PREPARATION_VALIDATION_V0_2_RULE_VERSION = (
    "final_preparation_validation_v0.2"
)


# ============================================================
# LEGACY CHECKS REPLACED BY LINEAGE-AWARE VALIDATION
# ============================================================


LEGACY_OUTPUT_SCOPE_CHECK_CODES = {
    "transform_dataset_scope",
    "combine_dataset_scope",
}


# ============================================================
# CHECK
# ============================================================


class FinalPreparationValidationV02Check(
    BaseModel,
):
    code: str

    passed: bool

    message: str

    source: str


# ============================================================
# REPORT
# ============================================================


class FinalPreparationValidationV02Report(
    BaseModel,
):
    workflow_id: str

    session_revision: int

    passed: bool

    preparation_root_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    analysis_output_dataset_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    check_count: int

    passed_check_count: int

    failed_check_count: int

    checks: List[
        FinalPreparationValidationV02Check
    ] = Field(
        default_factory=list
    )

    blocking_reasons: List[
        str
    ] = Field(
        default_factory=list
    )

    analysis_output_selection: (
        AnalysisOutputSelectionReport
        |
        None
    ) = None

    notes: List[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        FINAL_PREPARATION_VALIDATION_V0_2_RULE_VERSION
    )


# ============================================================
# ERROR
# ============================================================


class FinalPreparationValidationV02BlockedError(
    RuntimeError,
):
    def __init__(
        self,
        *,
        report: FinalPreparationValidationV02Report,
    ) -> None:
        self.report = (
            report
        )

        super().__init__(
            (
                "Final Preparation Validation v0.2 "
                "is blocked. "
                f"workflow_id={report.workflow_id}, "
                "failed_check_count="
                f"{report.failed_check_count}"
            )
        )


# ============================================================
# CHECK BUILDER
# ============================================================


def _check(
    *,
    code: str,
    passed: bool,
    success_message: str,
    failure_message: str,
    source: str,
) -> FinalPreparationValidationV02Check:
    return (
        FinalPreparationValidationV02Check(
            code=
                code,

            passed=
                passed,

            message=(
                success_message

                if passed

                else
                failure_message
            ),

            source=
                source,
        )
    )


# ============================================================
# LEGACY PREPARATION CHECKS
# ============================================================


def _legacy_preparation_checks(
    session: PreparationSessionView,
) -> List[
    FinalPreparationValidationV02Check
]:
    """
    Reuse Final Validation v0.1 for the Preparation-root
    contract.

    v0.1 remains authoritative for:

    - selected Preparation roots;
    - IMPORT;
    - UNDERSTAND;
    - QUALITY;
    - CLEAN;
    - stage existence;
    - stage resolution;
    - explicit deterministic cleaning evaluation.

    Two v0.1 checks are intentionally retired here:

        transform_dataset_scope
        combine_dataset_scope

    A derived TRANSFORM or COMBINE artifact does not need to
    have the same dataset_id as its Preparation roots.

    Its legitimacy is now proven through Artifact Store
    lineage instead.
    """

    legacy_report = (
        evaluate_final_preparation_validation_v0_1(
            session
        )
    )


    output: List[
        FinalPreparationValidationV02Check
    ] = []


    for legacy_check in (
        legacy_report.checks
    ):
        if (
            legacy_check.code
            in
            LEGACY_OUTPUT_SCOPE_CHECK_CODES
        ):
            continue


        output.append(
            FinalPreparationValidationV02Check(
                code=
                    legacy_check.code,

                passed=
                    legacy_check.passed,

                message=
                    legacy_check.message,

                source=
                    "final_validation_v0.1",
            )
        )


    return (
        output
    )


# ============================================================
# SUPERSEDED OUTPUT DETECTION
# ============================================================


def _superseded_selected_output_ids(
    *,
    selection_report:
        AnalysisOutputSelectionReport,

    selected_output_ids:
        List[
            str
        ],
) -> List[
    str
]:
    """
    Return selected analytical outputs that are no longer
    terminal in the current server-owned artifact graph.

    An artifact is superseded when another current artifact
    names it as a parent.

    CLEAN / TRANSFORM may materialize in place:

        dataset_id = "orders"
        parent_dataset_ids = ["orders"]

    That self-parent is provenance evidence and MUST NOT make
    the artifact superseded.

    Example:

        customers
            \
        transactions
              \
               combined_sales

    Once combined_sales exists, transactions and customers
    cannot be certified as final outputs if they are direct
    ancestors of a newer materialized artifact.

    This check intentionally uses the complete current
    server-owned candidate graph, not browser state.
    """

    used_as_parent = set()


    for candidate in (
        selection_report.candidates
    ):
        for parent_dataset_id in (
            candidate.parent_dataset_ids
        ):
            if (
                parent_dataset_id
                ==
                candidate.dataset_id
            ):
                continue


            used_as_parent.add(
                parent_dataset_id
            )


    return sorted(
        dataset_id

        for dataset_id
        in selected_output_ids

        if (
            dataset_id
            in
            used_as_parent
        )
    )


# ============================================================
# EVALUATE
# ============================================================


def evaluate_final_preparation_validation_v0_2(
    session: PreparationSessionView,
) -> FinalPreparationValidationV02Report:
    """
    Final Preparation gate with explicit separation between:

    1. Preparation root datasets
       session.selected_analysis_dataset_ids

    2. Final analytical outputs
       session.analysis_output_dataset_ids

    The root scope is checked using Final Validation v0.1.

    Final outputs are independently revalidated against the
    current Preparation Artifact Store and lineage.

    Final outputs must also belong to the current terminal
    materialized frontier. An artifact superseded by a newer
    descendant cannot be certified as a final analytical
    output.

    This function NEVER:

    - mutates PreparationSession;
    - mutates PreparationArtifactStore;
    - marks VALIDATE as PASSED;
    - authorizes ANALYZE.
    """

    preparation_root_ids = list(
        session
        .selected_analysis_dataset_ids
    )


    analysis_output_ids = list(
        session
        .analysis_output_dataset_ids
    )


    checks = (
        _legacy_preparation_checks(
            session
        )
    )


    # ========================================================
    # ANALYSIS OUTPUT SELECTION EXISTS
    # ========================================================


    outputs_selected = (
        len(
            analysis_output_ids
        )
        >
        0
    )


    checks.append(
        _check(
            code=
                "analysis_outputs_selected",

            passed=
                outputs_selected,

            success_message=(
                "At least one final analytical output "
                "is selected."
            ),

            failure_message=(
                "No final analytical output has been "
                "selected."
            ),

            source=
                "analysis_output_selection_v0.1",
        )
    )


    # ========================================================
    # REVALIDATE CURRENT ARTIFACTS + LINEAGE
    # ========================================================


    selection_report = (
        evaluate_analysis_output_selection(
            session=
                session,

            requested_dataset_ids=
                analysis_output_ids,
        )
    )


    outputs_still_authorized = (
        selection_report.valid
    )


    if (
        outputs_still_authorized
    ):
        lineage_failure_message = (
            "Final analytical output lineage is valid."
        )

    else:
        details = (
            "; ".join(
                selection_report
                .blocking_reasons
            )
        )


        lineage_failure_message = (
            (
                "Final analytical outputs are no longer "
                "authorized by current server-owned "
                "artifacts and lineage."
            )

            if not details

            else
            (
                "Final analytical outputs are no longer "
                "authorized by current server-owned "
                "artifacts and lineage. "
                f"{details}"
            )
        )


    checks.append(
        _check(
            code=
                "analysis_output_lineage_valid",

            passed=
                outputs_still_authorized,

            success_message=(
                "Final analytical outputs are backed by "
                "current server-owned artifacts and "
                "authorized lineage."
            ),

            failure_message=
                lineage_failure_message,

            source=
                "analysis_output_selection_v0.1",
        )
    )


    # ========================================================
    # TERMINAL OUTPUT FRONTIER
    # ========================================================
    #
    # Existing lineage validation answers:
    #
    #     "Is this artifact legitimate?"
    #
    # It does NOT answer:
    #
    #     "Is this still the latest materialized artifact?"
    #
    # Therefore a valid root / CLEAN / TRANSFORM artifact can
    # remain lineage-valid after COMBINE created a descendant.
    #
    # Final VALIDATE must fail closed in that situation.
    # ========================================================


    superseded_output_ids = (
        _superseded_selected_output_ids(
            selection_report=
                selection_report,

            selected_output_ids=
                analysis_output_ids,
        )
    )


    outputs_are_terminal = (
        outputs_still_authorized
        and
        not superseded_output_ids
    )


    terminal_failure_message = (
        (
            "One or more selected analytical outputs have "
            "been superseded by newer materialized "
            "Preparation artifacts."
        )

        if not superseded_output_ids

        else
        (
            "One or more selected analytical outputs have "
            "been superseded by newer materialized "
            "Preparation artifacts. "
            "superseded_dataset_ids="
            f"{superseded_output_ids}"
        )
    )


    checks.append(
        _check(
            code=
                "analysis_outputs_terminal",

            passed=
                outputs_are_terminal,

            success_message=(
                "Every selected analytical output is "
                "terminal in the current server-owned "
                "Preparation artifact graph."
            ),

            failure_message=
                terminal_failure_message,

            source=
                "final_preparation_validation_v0.2",
        )
    )


    # ========================================================
    # EXACT SESSION / SELECTION RECONCILIATION
    # ========================================================


    selection_matches_session = (
        outputs_still_authorized
        and
        (
            selection_report
            .selected_analysis_output_dataset_ids
            ==
            analysis_output_ids
        )
    )


    checks.append(
        _check(
            code=
                "analysis_output_scope_reconciled",

            passed=
                selection_matches_session,

            success_message=(
                "The validated analytical output scope "
                "exactly matches the server-owned session "
                "selection."
            ),

            failure_message=(
                "The current analytical output validation "
                "does not exactly match the server-owned "
                "session selection."
            ),

            source=
                "final_preparation_validation_v0.2",
        )
    )


    # ========================================================
    # ROOT LINEAGE RECONCILIATION
    # ========================================================


    analysis_output_set = set(
        analysis_output_ids
    )


    selected_candidates = [
        candidate

        for candidate
        in selection_report.candidates

        if (
            candidate.dataset_id
            in
            analysis_output_set
        )
    ]


    output_lineage_roots = set()


    for candidate in (
        selected_candidates
    ):
        output_lineage_roots.update(
            candidate
            .lineage_root_dataset_ids
        )


    preparation_root_set = set(
        preparation_root_ids
    )


    roots_authorized = (
        outputs_still_authorized
        and
        output_lineage_roots
        .issubset(
            preparation_root_set
        )
        and
        bool(
            output_lineage_roots
        )
    )


    checks.append(
        _check(
            code=
                "analysis_output_roots_authorized",

            passed=
                roots_authorized,

            success_message=(
                "Every selected analytical output resolves "
                "to authorized Preparation roots."
            ),

            failure_message=(
                "One or more selected analytical outputs "
                "do not resolve exclusively to authorized "
                "Preparation roots."
            ),

            source=
                "final_preparation_validation_v0.2",
        )
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================


    failed_checks = [
        check

        for check
        in checks

        if not (
            check.passed
        )
    ]


    blocking_reasons = [
        check.message

        for check
        in failed_checks
    ]


    passed = (
        len(
            failed_checks
        )
        ==
        0
    )


    return (
        FinalPreparationValidationV02Report(
            workflow_id=
                session.workflow_id,

            session_revision=
                session.revision,

            passed=
                passed,

            preparation_root_dataset_ids=
                preparation_root_ids,

            analysis_output_dataset_ids=
                analysis_output_ids,

            check_count=
                len(
                    checks
                ),

            passed_check_count=
                sum(
                    1

                    for check
                    in checks

                    if (
                        check.passed
                    )
                ),

            failed_check_count=
                len(
                    failed_checks
                ),

            checks=
                checks,

            blocking_reasons=
                blocking_reasons,

            analysis_output_selection=
                selection_report,

            notes=[
                (
                    "Final Preparation Validation v0.2 "
                    "separates Preparation roots from final "
                    "analytical outputs."
                ),

                (
                    "IMPORT, UNDERSTAND, QUALITY and CLEAN "
                    "continue to be evaluated against the "
                    "Preparation-root scope."
                ),

                (
                    "The legacy TRANSFORM and COMBINE "
                    "dataset-scope equality checks are "
                    "replaced by server-owned Artifact Store "
                    "lineage validation."
                ),

                (
                    "Final analytical outputs are "
                    "revalidated at validation time rather "
                    "than trusting an earlier selection "
                    "decision."
                ),

                (
                    "A selected analytical output must still "
                    "be terminal in the current materialized "
                    "artifact graph. A superseded ancestor "
                    "cannot pass VALIDATE."
                ),

                (
                    "Self-parent lineage produced by "
                    "in-place CLEAN or TRANSFORM "
                    "materialization does not supersede the "
                    "current artifact."
                ),

                (
                    "This evaluator does not mutate the "
                    "VALIDATE stage. Commit remains a "
                    "separate operation."
                ),
            ],

            rule_version=(
                FINAL_PREPARATION_VALIDATION_V0_2_RULE_VERSION
            ),
        )
    )


# ============================================================
# REQUIRE
# ============================================================


def require_final_preparation_validation_v0_2(
    session: PreparationSessionView,
) -> FinalPreparationValidationV02Report:
    report = (
        evaluate_final_preparation_validation_v0_2(
            session
        )
    )


    if not (
        report.passed
    ):
        raise (
            FinalPreparationValidationV02BlockedError(
                report=
                    report
            )
        )


    return (
        report
    )