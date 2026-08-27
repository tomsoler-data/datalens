import type {
  SemanticReviewReportView,
} from "./preparationTypes";


const API_URL =
  process.env.NEXT_PUBLIC_DATALENS_API_URL ??
  "http://127.0.0.1:8000";


export class SemanticReviewApiError extends Error {
  readonly status: number;
  readonly detail: unknown;


  constructor(
    message: string,
    status: number,
    detail: unknown
  ) {
    super(message);

    this.name =
      "SemanticReviewApiError";

    this.status =
      status;

    this.detail =
      detail;
  }
}


export async function requestSemanticReview({
  datasetFiles,
  workflowId,
  approvedCleaningActionIds,
  model,
}: {
  datasetFiles: File[];
  workflowId: string;
  approvedCleaningActionIds: string[];
  model: string;
}): Promise<
  SemanticReviewReportView
> {
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
    "workflow_id",
    workflowId
  );


  if (
    approvedCleaningActionIds.length >
      0
  ) {
    formData.append(
      "approved_action_ids_json",
      JSON.stringify(
        approvedCleaningActionIds
      )
    );
  }


  formData.append(
    "model",
    model
  );


  const response =
    await fetch(
      `${API_URL}/preparation/semantic-review`,
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


    throw (
      new SemanticReviewApiError(
        detail,

        response.status,

        payload
      )
    );
  }


  return (
    payload as
      SemanticReviewReportView
  );
}
