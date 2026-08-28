import type {
  PreparationSubstep,
} from "./PreparationSubstepNavigation";


const ACTIVE_PREPARATION_SUBSTEP_STORAGE_PREFIX =
  "datalens.activePreparationSubstep.v0.1:";


function activePreparationSubstepStorageKey(
  workflowId:
    string
): string {
  return (
    ACTIVE_PREPARATION_SUBSTEP_STORAGE_PREFIX +
    workflowId.trim()
  );
}


export function readActivePreparationSubstep(
  workflowId:
    string
): PreparationSubstep |
  null {
  if (
    typeof window ===
      "undefined"
  ) {
    return null;
  }


  const normalizedWorkflowId =
    workflowId.trim();


  if (
    !normalizedWorkflowId
  ) {
    return null;
  }


  try {
    const value =
      window.localStorage
        .getItem(
          activePreparationSubstepStorageKey(
            normalizedWorkflowId
          )
        )
        ?.trim() ??
      "";


    switch (
      value
    ) {
      case "understand":
      case "quality":
      case "cleaning":
      case "transform":
      case "combine":
      case "finalization":
        return value;

      default:
        return null;
    }
  } catch {
    return null;
  }
}


export function persistActivePreparationSubstep(
  workflowId:
    string,

  step:
    PreparationSubstep
): void {
  if (
    typeof window ===
      "undefined"
  ) {
    return;
  }


  const normalizedWorkflowId =
    workflowId.trim();


  if (
    !normalizedWorkflowId
  ) {
    return;
  }


  try {
    window.localStorage
      .setItem(
        activePreparationSubstepStorageKey(
          normalizedWorkflowId
        ),
        step
      );
  } catch {
    /*
     * Optional browser presentation state only.
     *
     * Preparation workflow state remains server-owned.
     */
  }
}
