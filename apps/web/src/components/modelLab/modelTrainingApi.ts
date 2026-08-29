import type {
  ModelTrainingApiErrorDetail,
  ModelTrainingContextResponse,
  ModelTrainingRequest,
  ModelTrainingResult,
} from "./modelTrainingTypes";


const API_URL =
  process.env
    .NEXT_PUBLIC_DATALENS_API_URL ??
  "http://127.0.0.1:8000";


/* ============================================================
   ERROR
============================================================ */


export class ModelTrainingApiError
  extends Error {
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
      "ModelTrainingApiError";

    this.status =
      status;

    this.detail =
      detail;
  }
}


/* ============================================================
   HELPERS
============================================================ */


function normalizeIdentifier(
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
    throw (
      new Error(
        `${fieldName} est requis.`
      )
    );
  }


  return normalized;
}


function structuredErrorDetail(
  payload:
    unknown
): ModelTrainingApiErrorDetail | null {
  if (
    !payload ||
    typeof payload !==
      "object" ||
    Array.isArray(
      payload
    )
  ) {
    return null;
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
    detail as
      Record<
        string,
        unknown
      >;


  if (
    typeof detailRecord.error !==
      "string" ||
    typeof detailRecord.message !==
      "string" ||
    typeof detailRecord.retryable !==
      "boolean" ||
    detailRecord.api_version !==
      "model_training_api_v0.1"
  ) {
    return null;
  }


  return (
    detailRecord as
      ModelTrainingApiErrorDetail
  );
}


function errorMessageFromPayload(
  payload:
    unknown
): string {
  const structured =
    structuredErrorDetail(
      payload
    );


  if (
    structured
  ) {
    return (
      structured.message
    );
  }


  if (
    typeof payload ===
      "string"
  ) {
    return payload;
  }


  if (
    payload &&
    typeof payload ===
      "object" &&
    !Array.isArray(
      payload
    )
  ) {
    const record =
      payload as
        Record<
          string,
          unknown
        >;


    if (
      typeof record.detail ===
        "string"
    ) {
      return record.detail;
    }


    if (
      typeof record.message ===
        "string"
    ) {
      return record.message;
    }
  }


  return (
    "La requête Model Training a échoué."
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
      new ModelTrainingApiError(
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


/* ============================================================
   GET TRAINING CONTEXT
============================================================ */


export async function getModelTrainingContext(
  workflowId:
    string,

  signal?:
    AbortSignal
): Promise<
  ModelTrainingContextResponse
> {
  const workflowIdNormalized =
    normalizeIdentifier(
      workflowId,
      "workflow_id"
    );


  const query =
    new URLSearchParams({
      workflow_id:
        workflowIdNormalized,
    });


  const response =
    await fetch(
      (
        `${API_URL}/model-training/context?` +
        query.toString()
      ),
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
      ModelTrainingContextResponse
    >(
      response
    )
  );
}


/* ============================================================
   TRAIN MODEL
============================================================ */


export async function trainModel(
  request:
    ModelTrainingRequest,

  signal?:
    AbortSignal
): Promise<
  ModelTrainingResult
> {
  normalizeIdentifier(
    request
      .training
      .workflow_id,

    "workflow_id"
  );


  normalizeIdentifier(
    request
      .training
      .dataset_id,

    "dataset_id"
  );


  normalizeIdentifier(
    request
      .training
      .target_column,

    "target_column"
  );


  normalizeIdentifier(
    request
      .training
      .estimator_key,

    "estimator_key"
  );


  if (
    request
      .training
      .feature_columns
      .length ===
      0
  ) {
    throw (
      new Error(
        "Au moins une variable explicative est requise."
      )
    );
  }


  if (
    !Number.isInteger(
      request
        .expected_preparation_session_revision
    ) ||
    request
      .expected_preparation_session_revision <
      0
  ) {
    throw (
      new Error(
        "La révision Preparation est invalide."
      )
    );
  }


  const response =
    await fetch(
      `${API_URL}/model-training/train`,
      {
        method:
          "POST",

        headers:
          {
            "Content-Type":
              "application/json",
          },

        body:
          JSON.stringify(
            request
          ),

        signal,
      }
    );


  return (
    requireSuccessfulJson<
      ModelTrainingResult
    >(
      response
    )
  );
}
