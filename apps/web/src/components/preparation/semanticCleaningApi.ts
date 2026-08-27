import type {
  SemanticCleaningApplyResponseView,
  SemanticCleaningChoiceView,
} from "./preparationTypes";


const API_URL =
  process.env.NEXT_PUBLIC_DATALENS_API_URL ??
  "http://127.0.0.1:8000";


export class SemanticCleaningApiError extends Error {
  readonly status: number;
  readonly detail: unknown;


  constructor(
    message: string,
    status: number,
    detail: unknown
  ) {
    super(message);

    this.name =
      "SemanticCleaningApiError";

    this.status =
      status;

    this.detail =
      detail;
  }
}


export async function applySemanticCleaning({
  datasetFiles,
  workflowId,
  approvedCleaningActionIds,
  semanticDecisions,
  approvedSemanticChoices,
}: {
  datasetFiles: File[];
  workflowId: string;
  approvedCleaningActionIds: string[];
  semanticDecisions: unknown[];
  approvedSemanticChoices:
    SemanticCleaningChoiceView[];
}): Promise<
  SemanticCleaningApplyResponseView
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
    "semantic_decisions_json",
    JSON.stringify(
      semanticDecisions
    )
  );


  formData.append(
    "approved_semantic_choices_json",
    JSON.stringify(
      approvedSemanticChoices
    )
  );


  const response =
    await fetch(
      `${API_URL}/preparation/semantic-cleaning-apply`,
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
      new SemanticCleaningApiError(
        detail,

        response.status,

        payload
      )
    );
  }


  return (
    payload as
      SemanticCleaningApplyResponseView
  );
}
