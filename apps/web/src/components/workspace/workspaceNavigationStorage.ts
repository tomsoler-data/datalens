import type { WorkspaceStep }
  from "./workspaceNavigationTypes";


const ACTIVE_WORKSPACE_STEP_STORAGE_PREFIX =
  "datalens.activeWorkspaceStep.v0.2:";


function activeWorkspaceStepStorageKey(
  workflowId:
    string
): string {
  return (
    ACTIVE_WORKSPACE_STEP_STORAGE_PREFIX +
    workflowId.trim()
  );
}


export function readActiveWorkspaceStep(
  workflowId:
    string
): WorkspaceStep |
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
          activeWorkspaceStepStorageKey(
            normalizedWorkflowId
          )
        )
        ?.trim() ??
      "";


    switch (
      value
    ) {
      case "data":
      case "documents":
      case "preparation":
      case "analyses":
      case "report":
        return value;

      default:
        return null;
    }
  } catch {
    return null;
  }
}


export function persistActiveWorkspaceStep(
  workflowId:
    string,

  step:
    WorkspaceStep
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
        activeWorkspaceStepStorageKey(
          normalizedWorkflowId
        ),
        step
      );
  } catch {
    /*
     * Optional browser presentation state only.
     * Backend state remains authoritative.
     */
  }
}
