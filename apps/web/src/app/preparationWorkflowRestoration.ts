import type {
  PreparationSessionView,
  PreparationUiStateView,
} from "../components/preparation/preparationTypes";

import type {
  PreparationSubstep,
} from "../components/preparation/PreparationSubstepNavigation";

import {
  findPreparationStage,
  preparationStageResolved,
  preparationSubstepFromSession,
  requiresCombineDiscoveryBeforeValidation,
} from "../components/preparation/preparationWorkflowHelpers";

import {
  readActivePreparationSubstep,
} from "../components/preparation/preparationSubstepStorage";

import type {
  WorkspaceStep,
} from "../components/workspace/workspaceNavigationTypes";

import {
  readActiveWorkspaceStep,
} from "../components/workspace/workspaceNavigationStorage";


export function resolveRestoredWorkspaceStep(
  workflowId:
    string,

  readyForAnalysis:
    boolean
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


export function preparationSubstepAvailableForSession(
  session:
    PreparationSessionView,

  step:
    PreparationSubstep
): boolean {
  const understand =
    findPreparationStage(
      session,
      "understand"
    );

  const quality =
    findPreparationStage(
      session,
      "quality"
    );

  const clean =
    findPreparationStage(
      session,
      "clean"
    );

  const transform =
    findPreparationStage(
      session,
      "transform"
    );

  const combine =
    findPreparationStage(
      session,
      "combine"
    );

  const validate =
    findPreparationStage(
      session,
      "validate"
    );


  const understandResolved =
    preparationStageResolved(
      understand
    );

  const qualityResolved =
    preparationStageResolved(
      quality
    );

  const cleanResolved =
    preparationStageResolved(
      clean
    );

  const transformResolved =
    preparationStageResolved(
      transform
    );

  const combineResolved =
    preparationStageResolved(
      combine
    );


  const finalizationDone =
    session.snapshot
      .ready_for_analysis ===
      true ||
    validate?.status ===
      "passed";


  const combineDiscoveryPending =
    !finalizationDone &&
    requiresCombineDiscoveryBeforeValidation(
      session
    );


  switch (
    step
  ) {
    case "understand":
      return true;

    case "quality":
      return understandResolved;

    case "cleaning":
      return qualityResolved;

    case "transform":
      return cleanResolved;

    case "combine":
      return transformResolved;

    case "finalization":
      return (
        finalizationDone ||
        (
          combineResolved &&
          !combineDiscoveryPending
        ) ||
        (
          !combineDiscoveryPending &&
          session.snapshot
            .next_stage ===
            "validate"
        )
      );

    default:
      return false;
  }
}


export function resolveRestoredPreparationSubstep(
  workflowId:
    string,

  session:
    PreparationSessionView
): PreparationSubstep {
  const restoredSubstep =
    readActivePreparationSubstep(
      workflowId
    );


  if (
    restoredSubstep !==
      null &&
    preparationSubstepAvailableForSession(
      session,
      restoredSubstep
    )
  ) {
    return restoredSubstep;
  }


  return preparationSubstepFromSession(
    session
  );
}


export function deriveRestoredAppliedCleaningActionIds(
  restoredUiState:
    PreparationUiStateView
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
  restoredUiState:
    PreparationUiStateView,

  restoredAppliedCleaningActionIds:
    string[]
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
