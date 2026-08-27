import type {
  AnalysisFollowUpTurn,
  ReportAvailableAnalysisDetailView,
  ReportSelectionDetailsView,
} from "../components/analysis/analysisTypes";


export type ReportSelectionRestoration = {
  restoredInitialPrompt: ReportAvailableAnalysisDetailView | null;
  initialPromptIncludedInReport: boolean;
  restoredFollowUpTurns: AnalysisFollowUpTurn[];
};


export function deriveReportSelectionRestoration(
  selection: ReportSelectionDetailsView,
  availableAnalyses: ReportAvailableAnalysisDetailView[],
): ReportSelectionRestoration {
  const selectedIds =
    new Set(
      selection
        .analyses
        .map(
          detail =>
            detail
              .selection
              .analysis_id
        )
    );


  /*
   * Rebuild transient prompt history from the
   * authoritative server-owned AnalysisArtifact list.
   *
   * Before F5, follow-up turns exist in React.
   * After F5 they are reconstructed from persisted
   * server-owned analysis artifacts.
   */
  const restoredPromptAnalyses =
    availableAnalyses
      .filter(
        analysis =>
          analysis.executed &&
          (
            analysis.source_type ===
              "initial_request" ||
            analysis.source_type ===
              "follow_up_prompt"
          )
      )
      .slice()
      .sort(
        (
          left,
          right
        ) =>
          left
            .created_at_utc
            .localeCompare(
              right
                .created_at_utc
            )
      );


  const restoredInitialPrompt =
    restoredPromptAnalyses
      .find(
        analysis =>
          analysis.source_type ===
            "initial_request"
      ) ??
    null;


  const initialPromptIncludedInReport =
    restoredInitialPrompt
      ? selectedIds.has(
          restoredInitialPrompt
            .analysis_id
        )
      : false;


  const restoredFollowUpTurns =
    restoredPromptAnalyses
      .filter(
        analysis =>
          analysis.source_type ===
            "follow_up_prompt"
      )
      .map(
        analysis => ({
          id:
            analysis
              .analysis_id,

          objective:
            analysis
              .objective,

          report:
            analysis
              .pipeline_payload,

          included_in_report:
            selectedIds.has(
              analysis
                .analysis_id
            ),
        })
      );


  return {
    restoredInitialPrompt,
    initialPromptIncludedInReport,
    restoredFollowUpTurns,
  };
}
