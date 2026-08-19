import type {
  CreatePreparationSessionRequest,
  PreparationAnalysisOutputCandidatesResponse,
  PreparationCombineDiscoveryResponse,
  PreparationCombineExecutionResponse,
  PreparationIdentityContinueResponse,
  PreparationIdentityCreateSurrogateResponse,
  PreparationIdentityInspectResponse,
  PreparationOutputExplanationResponse,
  PreparationSessionCapabilities,
  PreparationSessionView,
} from "./preparationTypes";


const API_URL =
  process.env
    .NEXT_PUBLIC_DATALENS_API_URL ??
  "http://127.0.0.1:8000";


export class PreparationApiError extends Error {
  readonly status:
    number;

  readonly detail:
    unknown;


  constructor(
    message:
      string,

    status:
      number,

    detail:
      unknown
  ) {
    super(
      message
    );

    this.name =
      "PreparationApiError";

    this.status =
      status;

    this.detail =
      detail;
  }
}


function errorMessageFromPayload(
  payload:
    unknown
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
      payload as
        Record<
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
        detail as
          Record<
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
    "La requête de préparation a échoué."
  );
}


async function readJsonResponse(
  response:
    Response
): Promise<unknown> {
  const contentType =
    response.headers.get(
      "content-type"
    ) ??
    "";


  if (
    !contentType.includes(
      "application/json"
    )
  ) {
    const text =
      await response.text();


    return (
      text ||
      null
    );
  }


  return (
    await response.json()
  );
}


async function requireSuccessfulJson<
  T
>(
  response:
    Response
): Promise<T> {
  const payload =
    await readJsonResponse(
      response
    );


  if (
    !response.ok
  ) {
    throw (
      new PreparationApiError(
        errorMessageFromPayload(
          payload
        ),

        response.status,

        payload
      )
    );
  }


  return (
    payload as T
  );
}


function normalizeWorkflowId(
  workflowId:
    string
): string {
  const normalizedId =
    workflowId.trim();


  if (
    !normalizedId
  ) {
    throw (
      new Error(
        "workflow_id est requis."
      )
    );
  }


  return normalizedId;
}


function normalizeDatasetIds(
  datasetIds:
    string[]
): string[] {
  const normalizedIds =
    Array.from(
      new Set(
        datasetIds
          .map(
            (
              datasetId
            ) =>
              datasetId.trim()
          )
          .filter(
            Boolean
          )
      )
    );


  if (
    normalizedIds.length ===
      0
  ) {
    throw (
      new Error(
        "Au moins un dataset doit être sélectionné."
      )
    );
  }


  return normalizedIds;
}


export async function getPreparationSessionCapabilities(
  signal?:
    AbortSignal
): Promise<
  PreparationSessionCapabilities
> {
  const response =
    await fetch(
      `${API_URL}/preparation/sessions/capabilities`,
      {
        method:
          "GET",

        cache:
          "no-store",

        signal,
      }
    );


  return (
    requireSuccessfulJson<
      PreparationSessionCapabilities
    >(
      response
    )
  );
}


export async function createPreparationSession(
  datasetIds:
    string[],

  signal?:
    AbortSignal
): Promise<
  PreparationSessionView
> {
  const normalizedIds =
    normalizeDatasetIds(
      datasetIds
    );


  const body:
    CreatePreparationSessionRequest = {
      selected_analysis_dataset_ids:
        normalizedIds,
    };


  const response =
    await fetch(
      `${API_URL}/preparation/sessions`,
      {
        method:
          "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body:
          JSON.stringify(
            body
          ),

        signal,
      }
    );


  return (
    requireSuccessfulJson<
      PreparationSessionView
    >(
      response
    )
  );
}


export async function getPreparationSession(
  workflowId:
    string,

  signal?:
    AbortSignal
): Promise<
  PreparationSessionView
> {
  const normalizedId =
    normalizeWorkflowId(
      workflowId
    );


  const response =
    await fetch(
      `${
        API_URL
      }/preparation/sessions/${
        encodeURIComponent(
          normalizedId
        )
      }`,
      {
        method:
          "GET",

        cache:
          "no-store",

        signal,
      }
    );


  return (
    requireSuccessfulJson<
      PreparationSessionView
    >(
      response
    )
  );
}


export async function getPreparationAnalysisOutputCandidates(
  workflowId:
    string,

  signal?:
    AbortSignal
): Promise<
  PreparationAnalysisOutputCandidatesResponse
> {
  const normalizedId =
    normalizeWorkflowId(
      workflowId
    );


  const response =
    await fetch(
      `${
        API_URL
      }/preparation/sessions/${
        encodeURIComponent(
          normalizedId
        )
      }/analysis-output-candidates`,
      {
        method:
          "GET",

        cache:
          "no-store",

        signal,
      }
    );


  return (
    requireSuccessfulJson<
      PreparationAnalysisOutputCandidatesResponse
    >(
      response
    )
  );
}


export async function selectPreparationAnalysisOutput(
  workflowId:
    string,

  datasetIds:
    string[],

  signal?:
    AbortSignal
): Promise<
  PreparationSessionView
> {
  const normalizedWorkflowId =
    normalizeWorkflowId(
      workflowId
    );


  const normalizedDatasetIds =
    normalizeDatasetIds(
      datasetIds
    );


  const response =
    await fetch(
      `${API_URL}/preparation/analysis-output`,
      {
        method:
          "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body:
          JSON.stringify(
            {
              workflow_id:
                normalizedWorkflowId,

              dataset_ids:
                normalizedDatasetIds,
            }
          ),

        signal,
      }
    );


  return (
    requireSuccessfulJson<
      PreparationSessionView
    >(
      response
    )
  );
}


export async function validatePreparationSession(
  workflowId:
    string,

  signal?:
    AbortSignal
): Promise<
  PreparationSessionView
> {
  const normalizedId =
    normalizeWorkflowId(
      workflowId
    );


  const response =
    await fetch(
      `${API_URL}/preparation/validate`,
      {
        method:
          "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body:
          JSON.stringify(
            {
              workflow_id:
                normalizedId,
            }
          ),

        signal,
      }
    );


  return (
    requireSuccessfulJson<
      PreparationSessionView
    >(
      response
    )
  );
}


// ============================================================
// CONTROLLED COMBINE WORKFLOW
// ============================================================


function normalizeCombineRequestId(
  requestId:
    string
): string {
  const normalizedId =
    requestId.trim();


  if (
    !normalizedId
  ) {
    throw (
      new Error(
        "request_id de jointure est requis."
      )
    );
  }


  return normalizedId;
}




// ============================================================
// DATASET IDENTITY
// ============================================================


export async function inspectPreparationIdentity(
  workflowId:
    string,

  datasetId:
    string,

  includeAi:
    boolean = true,

  signal?:
    AbortSignal
): Promise<
  PreparationIdentityInspectResponse
> {
  const normalizedWorkflowId =
    normalizeWorkflowId(
      workflowId
    );

  const normalizedDatasetId =
    datasetId.trim();


  if (
    !normalizedDatasetId
  ) {
    throw (
      new Error(
        "dataset_id est requis."
      )
    );
  }


  const response =
    await fetch(
      `${
        API_URL
      }/preparation/identity/inspect`,
      {
        method:
          "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body:
          JSON.stringify(
            {
              workflow_id:
                normalizedWorkflowId,

              dataset_id:
                normalizedDatasetId,

              include_ai:
                includeAi,
            }
          ),

        cache:
          "no-store",

        signal,
      }
    );


  return (
    requireSuccessfulJson<
      PreparationIdentityInspectResponse
    >(
      response
    )
  );
}


export async function continuePreparationWithoutSurrogate(
  workflowId:
    string,

  datasetId:
    string,

  requestId:
    string,

  signal?:
    AbortSignal
): Promise<
  PreparationIdentityContinueResponse
> {
  const normalizedWorkflowId =
    normalizeWorkflowId(
      workflowId
    );

  const normalizedDatasetId =
    datasetId.trim();

  const normalizedRequestId =
    requestId.trim();


  if (
    !normalizedDatasetId
  ) {
    throw (
      new Error(
        "dataset_id est requis."
      )
    );
  }


  if (
    !normalizedRequestId
  ) {
    throw (
      new Error(
        "request_id est requis."
      )
    );
  }


  const response =
    await fetch(
      `${
        API_URL
      }/preparation/identity/continue`,
      {
        method:
          "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body:
          JSON.stringify(
            {
              workflow_id:
                normalizedWorkflowId,

              dataset_id:
                normalizedDatasetId,

              request_id:
                normalizedRequestId,
            }
          ),

        cache:
          "no-store",

        signal,
      }
    );


  return (
    requireSuccessfulJson<
      PreparationIdentityContinueResponse
    >(
      response
    )
  );
}


export async function createPreparationSurrogateKey(
  workflowId:
    string,

  datasetId:
    string,

  requestId:
    string,

  signal?:
    AbortSignal
): Promise<
  PreparationIdentityCreateSurrogateResponse
> {
  const normalizedWorkflowId =
    normalizeWorkflowId(
      workflowId
    );

  const normalizedDatasetId =
    datasetId.trim();

  const normalizedRequestId =
    requestId.trim();


  if (
    !normalizedDatasetId
  ) {
    throw (
      new Error(
        "dataset_id est requis."
      )
    );
  }


  if (
    !normalizedRequestId
  ) {
    throw (
      new Error(
        "request_id est requis."
      )
    );
  }


  const response =
    await fetch(
      `${
        API_URL
      }/preparation/identity/create-surrogate`,
      {
        method:
          "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body:
          JSON.stringify(
            {
              workflow_id:
                normalizedWorkflowId,

              dataset_id:
                normalizedDatasetId,

              request_id:
                normalizedRequestId,
            }
          ),

        cache:
          "no-store",

        signal,
      }
    );


  return (
    requireSuccessfulJson<
      PreparationIdentityCreateSurrogateResponse
    >(
      response
    )
  );
}




// ============================================================
// ANALYSIS OUTPUT EXPLANATION
// ============================================================


export async function explainPreparationAnalysisOutput(
  workflowId:
    string,

  datasetId:
    string,

  includeAi:
    boolean = true,

  signal?:
    AbortSignal
): Promise<
  PreparationOutputExplanationResponse
> {
  const normalizedWorkflowId =
    normalizeWorkflowId(
      workflowId
    );

  const normalizedDatasetId =
    datasetId.trim();


  if (
    !normalizedDatasetId
  ) {
    throw (
      new Error(
        "dataset_id est requis."
      )
    );
  }


  const response =
    await fetch(
      `${
        API_URL
      }/preparation/analysis-output/explain`,
      {
        method:
          "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body:
          JSON.stringify(
            {
              workflow_id:
                normalizedWorkflowId,

              dataset_id:
                normalizedDatasetId,

              include_ai:
                includeAi,
            }
          ),

        cache:
          "no-store",

        signal,
      }
    );


  return (
    requireSuccessfulJson<
      PreparationOutputExplanationResponse
    >(
      response
    )
  );
}


export async function discoverPreparationCombine(
  workflowId:
    string,

  signal?:
    AbortSignal
): Promise<
  PreparationCombineDiscoveryResponse
> {
  const normalizedId =
    normalizeWorkflowId(
      workflowId
    );


  const response =
    await fetch(
      `${API_URL}/preparation/combine/discover`,
      {
        method:
          "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body:
          JSON.stringify(
            {
              workflow_id:
                normalizedId,
            }
          ),

        signal,
      }
    );


  return (
    requireSuccessfulJson<
      PreparationCombineDiscoveryResponse
    >(
      response
    )
  );
}


export async function approvePreparationCombine(
  workflowId:
    string,

  requestId:
    string,

  comment?:
    string,

  signal?:
    AbortSignal
): Promise<
  PreparationCombineExecutionResponse
> {
  const normalizedWorkflowId =
    normalizeWorkflowId(
      workflowId
    );

  const normalizedRequestId =
    normalizeCombineRequestId(
      requestId
    );

  const normalizedComment =
    comment
      ?.trim() ??
    "";


  const response =
    await fetch(
      `${API_URL}/preparation/combine/approve`,
      {
        method:
          "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body:
          JSON.stringify(
            {
              workflow_id:
                normalizedWorkflowId,

              request_id:
                normalizedRequestId,

              ...(
                normalizedComment
                  ? {
                      comment:
                        normalizedComment,
                    }
                  : {}
              ),
            }
          ),

        signal,
      }
    );


  return (
    requireSuccessfulJson<
      PreparationCombineExecutionResponse
    >(
      response
    )
  );
}

