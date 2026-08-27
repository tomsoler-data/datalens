const ACTIVE_PREPARATION_WORKFLOW_STORAGE_KEY =
  "datalens.activePreparationWorkflow.v0.1";


export function readActivePreparationWorkflowId():
  string |
  null {
  if (
    typeof window ===
      "undefined"
  ) {
    return null;
  }


  try {
    const workflowId =
      window.localStorage
        .getItem(
          ACTIVE_PREPARATION_WORKFLOW_STORAGE_KEY
        )
        ?.trim() ??
      "";


    return (
      workflowId ||
      null
    );
  } catch {
    return null;
  }
}


export function persistActivePreparationWorkflowId(
  workflowId:
    string
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
        ACTIVE_PREPARATION_WORKFLOW_STORAGE_KEY,
        normalizedWorkflowId
      );
  } catch {
    // Browser storage is an optional convenience only.
    // The backend Preparation store remains authoritative.
  }
}


export function clearActivePreparationWorkflowId(): void {
  if (
    typeof window ===
      "undefined"
  ) {
    return;
  }


  try {
    window.localStorage
      .removeItem(
        ACTIVE_PREPARATION_WORKFLOW_STORAGE_KEY
      );
  } catch {
    // Ignore unavailable browser storage.
  }
}
