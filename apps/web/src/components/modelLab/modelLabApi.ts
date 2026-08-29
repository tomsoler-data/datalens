import type {
  ModelLabApiErrorDetail,
  ModelLabEvaluationOptions,
  ModelLabEvaluationSummary,
  ModelLabModelDetail,
  ModelLabModelHealthSummary,
  ModelLabModelListResponse,
  ModelLabMonitoringAlertDecision,
  ModelLabPredictionRow,
  ModelLabPredictResponse,
} from "./modelLabTypes";


const API_URL =
  process.env
    .NEXT_PUBLIC_DATALENS_API_URL ??
  "http://127.0.0.1:8000";


export const MODEL_LAB_PREDICTION_MAX_ROWS =
  100;

export const MODEL_LAB_PREDICTION_MAX_COLUMNS =
  256;

export const MODEL_LAB_PREDICTION_MAX_CELLS =
  10_000;


/* ============================================================
   ERROR
============================================================ */


export class ModelLabApiError
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
      "ModelLabApiError";

    this.status =
      status;

    this.detail =
      detail;
  }
}


/* ============================================================
   RESPONSE HELPERS
============================================================ */


function structuredErrorDetail(
  payload:
    unknown
): ModelLabApiErrorDetail | null {
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
    typeof detailRecord.api_version !==
      "string" ||
    ![
      "model_lab_api_v0.1",
      "ml_model_health_api_v0.1",
      "ml_monitoring_alert_api_v0.1",
    ].includes(
      detailRecord.api_version
    )
  ) {
    return null;
  }


  return (
    detailRecord as
      ModelLabApiErrorDetail
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
    "La requête Model Lab a échoué."
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
      new ModelLabApiError(
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
   IDENTIFIERS
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


/* ============================================================
   MODEL LIST
============================================================ */


export async function listModelLabModels(
  workflowId:
    string,

  signal?:
    AbortSignal
): Promise<
  ModelLabModelListResponse
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
      `${API_URL}/model-lab/models?${query.toString()}`,
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
      ModelLabModelListResponse
    >(
      response
    )
  );
}


/* ============================================================
   MODEL DETAIL
============================================================ */


export async function getModelLabModelDetail(
  workflowId:
    string,

  modelId:
    string,

  signal?:
    AbortSignal
): Promise<
  ModelLabModelDetail
> {
  const workflowIdNormalized =
    normalizeIdentifier(
      workflowId,
      "workflow_id"
    );

  const modelIdNormalized =
    normalizeIdentifier(
      modelId,
      "model_id"
    );


  const query =
    new URLSearchParams({
      workflow_id:
        workflowIdNormalized,
    });


  const response =
    await fetch(
      (
        `${API_URL}/model-lab/models/` +
        `${encodeURIComponent(modelIdNormalized)}?` +
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
      ModelLabModelDetail
    >(
      response
    )
  );
}


/* ============================================================
   EVALUATION
============================================================ */


function normalizeDecisionThreshold(
  value:
    number
): number {
  if (
    !Number.isFinite(
      value
    ) ||
    value < 0 ||
    value > 1
  ) {
    throw (
      new Error(
        "Le seuil de décision doit être compris entre 0 et 1."
      )
    );
  }


  return value;
}


export async function evaluateModelLabModel(
  workflowId:
    string,

  modelId:
    string,

  options?:
    ModelLabEvaluationOptions,

  signal?:
    AbortSignal
): Promise<
  ModelLabEvaluationSummary
> {
  const workflowIdNormalized =
    normalizeIdentifier(
      workflowId,
      "workflow_id"
    );

  const modelIdNormalized =
    normalizeIdentifier(
      modelId,
      "model_id"
    );


  const threshold =
    options
      ?.decision_threshold
      ?.threshold;


  const evaluation =
    threshold ===
      undefined
      ? undefined
      : {
          decision_threshold:
            {
              threshold:
                normalizeDecisionThreshold(
                  threshold
                ),
            },
        };


  const requestBody = {
    workflow_id:
      workflowIdNormalized,

    model_id:
      modelIdNormalized,

    ...(
      evaluation
        ? {
            evaluation,
          }
        : {}
    ),
  };


  const response =
    await fetch(
      `${API_URL}/model-lab/evaluate`,
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
            requestBody
          ),

        signal,
      }
    );


  return (
    requireSuccessfulJson<
      ModelLabEvaluationSummary
    >(
      response
    )
  );
}


/* ============================================================
   MODEL HEALTH
============================================================ */


export async function getModelLabModelHealth(
  workflowId:
    string,

  modelId:
    string,

  signal?:
    AbortSignal
): Promise<
  ModelLabModelHealthSummary
> {
  const workflowIdNormalized =
    normalizeIdentifier(
      workflowId,
      "workflow_id"
    );

  const modelIdNormalized =
    normalizeIdentifier(
      modelId,
      "model_id"
    );


  const query =
    new URLSearchParams({
      workflow_id:
        workflowIdNormalized,
    });


  const response =
    await fetch(
      (
        `${API_URL}/ml-monitoring/models/` +
        `${encodeURIComponent(modelIdNormalized)}/health?` +
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
      ModelLabModelHealthSummary
    >(
      response
    )
  );
}


/* ============================================================
   MONITORING ALERT
============================================================ */


export async function getModelLabMonitoringAlert(
  workflowId:
    string,

  modelId:
    string,

  signal?:
    AbortSignal
): Promise<
  ModelLabMonitoringAlertDecision
> {
  const workflowIdNormalized =
    normalizeIdentifier(
      workflowId,
      "workflow_id"
    );

  const modelIdNormalized =
    normalizeIdentifier(
      modelId,
      "model_id"
    );


  const query =
    new URLSearchParams({
      workflow_id:
        workflowIdNormalized,
    });


  const response =
    await fetch(
      (
        `${API_URL}/ml-monitoring/models/` +
        `${encodeURIComponent(modelIdNormalized)}/alert?` +
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
      ModelLabMonitoringAlertDecision
    >(
      response
    )
  );
}


/* ============================================================
   PREDICTION INPUT
============================================================ */


function validatePredictionRows(
  rows:
    ModelLabPredictionRow[]
): void {
  if (
    rows.length ===
      0
  ) {
    throw (
      new Error(
        "Au moins une ligne de prédiction est requise."
      )
    );
  }


  if (
    rows.length >
      MODEL_LAB_PREDICTION_MAX_ROWS
  ) {
    throw (
      new Error(
        (
          "Le Model Lab accepte au maximum " +
          `${MODEL_LAB_PREDICTION_MAX_ROWS} lignes par prédiction.`
        )
      )
    );
  }


  let cellCount =
    0;


  rows.forEach(
    (
      row,
      rowIndex
    ) => {
      const columns =
        Object.keys(
          row
        );


      if (
        columns.length ===
          0
      ) {
        throw (
          new Error(
            (
              "Une ligne de prédiction ne peut pas être vide. " +
              `Ligne ${rowIndex + 1}.`
            )
          )
        );
      }


      if (
        columns.length >
          MODEL_LAB_PREDICTION_MAX_COLUMNS
      ) {
        throw (
          new Error(
            (
              "Une ligne dépasse la limite de " +
              `${MODEL_LAB_PREDICTION_MAX_COLUMNS} variables.`
            )
          )
        );
      }


      cellCount +=
        columns.length;


      for (
        const column
        of columns
      ) {
        if (
          column !==
            column.trim() ||
          !column.trim()
        ) {
          throw (
            new Error(
              "Les noms de variables de prédiction doivent être non vides et sans espaces périphériques."
            )
          );
        }


        const value =
          row[
            column
          ];


        if (
          typeof value ===
            "number" &&
          !Number.isFinite(
            value
          )
        ) {
          throw (
            new Error(
              (
                "Les valeurs numériques de prédiction " +
                "doivent être finies."
              )
            )
          );
        }
      }
    }
  );


  if (
    cellCount >
      MODEL_LAB_PREDICTION_MAX_CELLS
  ) {
    throw (
      new Error(
        (
          "La requête dépasse la limite Model Lab de " +
          `${MODEL_LAB_PREDICTION_MAX_CELLS} cellules.`
        )
      )
    );
  }
}


/* ============================================================
   PREDICT
============================================================ */


export async function predictModelLab(
  workflowId:
    string,

  modelId:
    string,

  rows:
    ModelLabPredictionRow[],

  signal?:
    AbortSignal
): Promise<
  ModelLabPredictResponse
> {
  const workflowIdNormalized =
    normalizeIdentifier(
      workflowId,
      "workflow_id"
    );

  const modelIdNormalized =
    normalizeIdentifier(
      modelId,
      "model_id"
    );


  validatePredictionRows(
    rows
  );


  const response =
    await fetch(
      `${API_URL}/model-lab/predict`,
      {
        method:
          "POST",

        headers:
          {
            "Content-Type":
              "application/json",
          },

        body:
          JSON.stringify({
            workflow_id:
              workflowIdNormalized,

            model_id:
              modelIdNormalized,

            rows,
          }),

        signal,
      }
    );


  return (
    requireSuccessfulJson<
      ModelLabPredictResponse
    >(
      response
    )
  );
}
