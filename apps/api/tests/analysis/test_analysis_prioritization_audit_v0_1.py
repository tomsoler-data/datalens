from __future__ import annotations

from app.analysis_prioritization import (
    ANALYSIS_PRIORITIZATION_AUDIT_RULE_VERSION,
    ANALYSIS_PRIORITIZATION_RULE_VERSION,
    ANALYTICAL_VALUE_GUARD_RULE_VERSION,
    AnalysisPrioritizationDecision,
    AnalysisPrioritizationReport,
    build_analysis_prioritization_audit,
)

from app.api.analysis_run import (
    RoutedUnifiedAnalysisReport,
)


def make_decision(
    *,
    analysis_id: str,
    decision: str,
    reason_code: str,
    title: str,
) -> AnalysisPrioritizationDecision:
    return AnalysisPrioritizationDecision(
        analysis_id=analysis_id,
        family="group_comparison",
        title=title,
        original_priority_score=70.0,
        execution_priority_score=70.0,
        decision=decision,
        reason_code=reason_code,
        reasons=[f"reason:{reason_code}"],
        variable_keys=["dataset:value"],
    )


def main() -> None:
    print("=== DATALENS ANALYSIS PRIORITIZATION AUDIT v0.1 ===")

    decisions = [
        make_decision(
            analysis_id="a-selected",
            decision="selected",
            reason_code="selected_by_priority",
            title="Prix selon catégorie",
        ),
        make_decision(
            analysis_id="a-label",
            decision="deferred",
            reason_code="record_label_dimension",
            title="Quantité selon prénom",
        ),
        make_decision(
            analysis_id="a-family",
            decision="deferred",
            reason_code="family_budget_exhausted",
            title="Prix selon canal",
        ),
        make_decision(
            analysis_id="a-identifier",
            decision="rejected",
            reason_code="identifier_misuse",
            title="Association avec identifiant",
        ),
    ]

    report = AnalysisPrioritizationReport(
        discovered_count=4,
        selected_count=1,
        deferred_count=2,
        rejected_count=1,
        selected_analysis_ids=["a-selected"],
        deferred_analysis_ids=["a-label", "a-family"],
        rejected_analysis_ids=["a-identifier"],
        decisions=decisions,
        selected_candidates=[],
        family_selected_counts={"group_comparison": 1},
        notes=[],
        rule_version=ANALYSIS_PRIORITIZATION_RULE_VERSION,
    )

    audit = build_analysis_prioritization_audit(report)

    assert audit.discovered_count == 4
    assert audit.selected_for_execution_count == 1
    assert audit.deferred_count == 2
    assert audit.rejected_count == 1
    print("Aggregate counters preserved: PASS")

    assert audit.decision_counts == {
        "deferred": 2,
        "rejected": 1,
        "selected": 1,
    }
    print("Decision counts deterministic: PASS")

    assert audit.non_execution_reason_counts == {
        "family_budget_exhausted": 1,
        "identifier_misuse": 1,
        "record_label_dimension": 1,
    }
    assert "selected_by_priority" not in audit.non_execution_reason_counts
    print("Non-execution reasons separated from selected reasons: PASS")

    assert len(audit.decisions) == 4
    assert audit.decisions[1].analysis_id == "a-label"
    print("Per-analysis audit decisions preserved: PASS")

    report.decisions[1].reasons.append("tampered")
    assert "tampered" not in audit.decisions[1].reasons
    print("Public audit defensively copies decisions: PASS")

    assert (
        audit.prioritization_rule_version
        == ANALYSIS_PRIORITIZATION_RULE_VERSION
    )
    assert (
        audit.analytical_value_guard_rule_version
        == ANALYTICAL_VALUE_GUARD_RULE_VERSION
    )
    assert (
        audit.audit_rule_version
        == ANALYSIS_PRIORITIZATION_AUDIT_RULE_VERSION
    )
    print("Rule versions exposed: PASS")

    assert (
        "prioritization_audit"
        in RoutedUnifiedAnalysisReport.model_fields
    )
    print("Routed HTTP report exposes prioritization audit: PASS")

    print(
        "Analysis Prioritization Audit version:",
        ANALYSIS_PRIORITIZATION_AUDIT_RULE_VERSION,
    )
    print("Analysis Prioritization Audit v0.1: PASS")


if __name__ == "__main__":
    main()
