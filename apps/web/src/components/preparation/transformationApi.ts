import type {
  PreparationTransformationApplyResponse,
  PreparationTransformationApprovalCommand,
  PreparationTransformationIntent,
  PreparationTransformationPlan,
} from "./transformationTypes";


const API_URL =
  process.env
    .NEXT_PUBLIC_DATALENS_API_URL ??
  "http://127.0.0.1:8000";


export class TransformationApiError extends Error {
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
      "TransformationApiError";

    this.status =
      status;

    this.detail =
      detail;
  }
}


function requiredText(
  value:
    string,

  fieldName:
    string
): string {
  const normalized =
    value.trim();


  if (
    !normalized
  ) {
    throw new Error(
      `${fieldName} est requis.`
    );
  }


  return normalized;
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
    !payload ||
    typeof payload !==
      "object" ||
    Array.isArray(
      payload
    )
  ) {
    return (
      "La requête de transformation a échoué."
    );
  }


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
      return detailRecord.message;
    }
  }


  if (
    typeof record.message ===
      "string"
  ) {
    return record.message;
  }


  return (
    "La requête de transformation a échoué."
  );
}


async function requireSuccessfulJson<
  T
>(
  response:
    Response
): Promise<T> {
  const contentType =
    response.headers.get(
      "content-type"
    ) ??
    "";


  const payload:
    unknown =
      contentType.includes(
        "application/json"
      )
        ? await response.json()
        : await response.text();


  if (
    !response.ok
  ) {
    throw new TransformationApiError(
      errorMessageFromPayload(
        payload
      ),

      response.status,

      payload
    );
  }


  return payload as T;
}


export async function buildPreparationTransformationPlan(
  workflowId:
    string,

  datasetId:
    string,

  intents:
    PreparationTransformationIntent[],

  signal?:
    AbortSignal
): Promise<
  PreparationTransformationPlan
> {
  const normalizedWorkflowId =
    requiredText(
      workflowId,
      "workflow_id"
    );


  const normalizedDatasetId =
    requiredText(
      datasetId,
      "dataset_id"
    );


  const response =
    await fetch(
      `${API_URL}/preparation/transformation-plan`,
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

              intents,
            }
          ),

        cache:
          "no-store",

        signal,
      }
    );


  return (
    requireSuccessfulJson<
      PreparationTransformationPlan
    >(
      response
    )
  );
}


export async function applyPreparationTransformation(
  workflowId:
    string,

  datasetId:
    string,

  intents:
    PreparationTransformationIntent[],

  approvalCommands:
    PreparationTransformationApprovalCommand[],

  signal?:
    AbortSignal
): Promise<
  PreparationTransformationApplyResponse
> {
  const normalizedWorkflowId =
    requiredText(
      workflowId,
      "workflow_id"
    );


  const normalizedDatasetId =
    requiredText(
      datasetId,
      "dataset_id"
    );


  const response =
    await fetch(
      `${API_URL}/preparation/transformation-apply`,
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

              intents,

              approval_commands:
                approvalCommands,
            }
          ),

        cache:
          "no-store",

        signal,
      }
    );


  return (
    requireSuccessfulJson<
      PreparationTransformationApplyResponse
    >(
      response
    )
  );
}