import type {
  CleaningApplyResponseView,
} from "./preparationTypes";


const API_URL =
  process.env.NEXT_PUBLIC_DATALENS_API_URL ??
  "http://127.0.0.1:8000";


export async function applyDeterministicCleaning({
  datasetFiles,
  workflowId,
  approvedCleaningActionIds,
  rejectedCleaningActionIds,
}: {
  datasetFiles: File[];
  workflowId: string;
  approvedCleaningActionIds: string[];
  rejectedCleaningActionIds: string[];
}): Promise<CleaningApplyResponseView> {
  const formData =
    new FormData();


  for (
    const file
    of datasetFiles
  ) {
    formData.append(
      "dataset_files",
      file
    );
  }


  formData.append(
    "approved_action_ids_json",
    JSON.stringify(
      approvedCleaningActionIds
    )
  );


  formData.append(
    "rejected_action_ids_json",
    JSON.stringify(
      rejectedCleaningActionIds
    )
  );


  formData.append(
    "workflow_id",
    workflowId
  );


  const response =
    await fetch(
      `${API_URL}/preparation/cleaning-apply`,
      {
        method:
          "POST",

        body:
          formData,
      }
    );


  const payload =
    await response.json();


  if (
    !response.ok
  ) {
    const detail =
      typeof payload.detail ===
        "string"
        ? payload.detail
        : JSON.stringify(
            payload.detail ??
            payload
          );


    throw new Error(
      detail
    );
  }


  return (
    payload as
      CleaningApplyResponseView
  );
}
