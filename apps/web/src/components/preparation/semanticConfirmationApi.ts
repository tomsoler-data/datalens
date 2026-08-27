import type {
  SemanticConfirmationReportView,
} from "./preparationTypes";

export type {
  SemanticConfirmationReportView,
} from "./preparationTypes";

export type SemanticManualResolutionView = {
  issue_id: string;
  note: string;
};




export type SemanticReviewConfirmationResponseView = {
  status: string;
  confirmation: SemanticConfirmationReportView;
  plan: unknown;
  execution: unknown;
};


const API_URL =
  process.env.NEXT_PUBLIC_DATALENS_API_URL ??
  "http://127.0.0.1:8000";


function semanticConfirmationFromPayload(
  payload: unknown
): SemanticConfirmationReportView |
  null {
  if (
    !payload ||
    typeof payload !==
      "object"
  ) {
    return null;
  }


  const outer =
    payload as Record<
      string,
      unknown
    >;


  const detail =
    outer.detail;


  if (
    !detail ||
    typeof detail !==
      "object" ||
    Array.isArray(
      detail
    )
  ) {
    return null;
  }


  const detailRecord =
    detail as Record<
      string,
      unknown
    >;


  const confirmation =
    detailRecord.confirmation;


  if (
    confirmation &&
    typeof confirmation ===
      "object"
  ) {
    return (
      confirmation as
        SemanticConfirmationReportView
    );
  }


  return null;
}


export class SemanticConfirmationApiError extends Error {
  readonly status: number;
  readonly detail: unknown;
  readonly confirmation:
    SemanticConfirmationReportView |
    null;



  constructor(
    message: string,
    status: number,
    detail: unknown
  ) {
    super(message);

    this.name =
      "SemanticConfirmationApiError";

    this.status =
      status;

    this.detail =
      detail;

    this.confirmation =
      semanticConfirmationFromPayload(
        detail
      );
  }
}


function errorMessageFromPayload(
  payload: unknown
): string {
  if (
    typeof payload ===
      "string"
  ) {
    return payload;
  }


  if (
    payload &&
    typeof payload ===
      "object"
  ) {
    const record =
      payload as Record<
        string,
        unknown
      >;


    const detail =
      record.detail;


    if (
      typeof detail ===
        "string"
    ) {
      return detail;
    }


    if (
      detail &&
      typeof detail ===
        "object" &&
      !Array.isArray(
        detail
      )
    ) {
      const detailRecord =
        detail as Record<
          string,
          unknown
        >;


      if (
        typeof detailRecord.message ===
          "string"
      ) {
        return (
          detailRecord.message
        );
      }
    }


    if (
      typeof record.message ===
        "string"
    ) {
      return record.message;
    }
  }


  return (
    "La confirmation sémantique a échoué."
  );
}


export async function confirmSemanticReview({
  datasetFiles,
  workflowId,
  semanticDecisions,
  confirmedIssueIds,
  approvedSemanticChoices,
  manualResolutions,
  approvedCleaningActionIds,
}: {
  datasetFiles: File[];
  workflowId: string;
  semanticDecisions: unknown[];
  confirmedIssueIds: string[];
  approvedSemanticChoices: unknown[];
  manualResolutions: SemanticManualResolutionView[];
  approvedCleaningActionIds: string[];
}): Promise<
  SemanticReviewConfirmationResponseView
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


  formData.append(
    "semantic_decisions_json",
    JSON.stringify(
      semanticDecisions
    )
  );


  formData.append(
    "confirmed_issue_ids_json",
    JSON.stringify(
      confirmedIssueIds
    )
  );


  formData.append(
    "approved_semantic_choices_json",
    JSON.stringify(
      approvedSemanticChoices
    )
  );


  formData.append(
    "manual_resolutions_json",
    JSON.stringify(
      manualResolutions
    )
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


  const response =
    await fetch(
      `${API_URL}/preparation/semantic-review-confirm`,
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
    throw (
      new SemanticConfirmationApiError(
        errorMessageFromPayload(
          payload
        ),

        response.status,

        payload
      )
    );
  }


  return (
    payload as
      SemanticReviewConfirmationResponseView
  );
}
