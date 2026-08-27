import type { PreparationUiStateView } from "../components/preparation/preparationTypes";
import type { WorkspaceStep } from "../components/workspace/workspaceNavigationTypes";
import { readActiveWorkspaceStep } from "../components/workspace/workspaceNavigationStorage";


export function resolveRestoredWorkspaceStep(
  workflowId: string,
  readyForAnalysis: boolean,
): WorkspaceStep {
  const restoredWorkspaceStep =
    readActiveWorkspaceStep(
      workflowId
    );


  return restoredWorkspaceStep ===
    "report"
      ? "analyses"
      : restoredWorkspaceStep ??
          (
            readyForAnalysis
              ? "analyses"
              : "preparation"
          );
}


export function deriveRestoredAppliedCleaningActionIds(
  restoredUiState: PreparationUiStateView,
): string[] {
  return (
    restoredUiState
      .cleaning_execution
      ?.action_results
      .filter(
        result =>
          result.status ===
            "applied"
      )
      .map(
        result =>
          result.action_id
      ) ?? []
  );
}


export function deriveRestoredSelectedCleaningActionIds(
  restoredUiState: PreparationUiStateView,
  restoredAppliedCleaningActionIds: string[],
): string[] {
  return restoredUiState
    .cleaning_execution
      ? restoredAppliedCleaningActionIds
      : (
          restoredUiState
            .cleaning_plan
            ?.actions
            .filter(
              action =>
                action.safe_candidate
            )
            .map(
              action =>
                action.action_id
            ) ?? []
        );
}
