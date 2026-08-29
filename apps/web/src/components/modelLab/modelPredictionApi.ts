import {
  ModelLabApiError,
} from "./modelLabApi";


const API_URL =
  process.env
    .NEXT_PUBLIC_DATALENS_API_URL ??
  "http://127.0.0.1:8000";


/* ============================================================
   CONTRACT
============================================================ */


export type ModelPredictionScalar =
  | string
  | number
  | boolean
  | null;


export type ModelPredictionRow =
  Record<
    string,
    ModelPredictionScalar
  >;


export type ModelPredictionRequest = {
  workflow_id:
    string;

  model_id:
    string;

  rows:
    ModelPredictionRow[];

  rule_version?:
    "model_lab_api_contract_v0.1";
};


export type ModelPredictionResponse = {
  workflow_id:
    string;

  model_id:
    string;

  problem_type:
    "regression"
    | "classification";

  target_column:
    string;

  prediction_count:
    number;

  predictions:
    ModelPredictionScalar[];

  method:
    "trusted_native_predict";

  rule_version:
    "model_lab_api_contract_v0.1";
};


/* ============================================================
   LIMITS
============================================================ */


const MAX_ROWS =
  100;


const MAX_COLUMNS =
  256;


const MAX_CELLS =
  10_000;


const MAX_STRING_LENGTH =
  10_000;


/* ============================================================
   VALIDATION
============================================================ */


function requiredIdentifier(
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


function validateScalar(
  value:
    ModelPredictionScalar
): void {
  if (
    typeof value ===
      "number"
    &&
    !Number.isFinite(
      value
    )
  ) {
    throw new Error(
      "Les valeurs numériques de prédiction doivent être finies."
    );
  }


  if (
    typeof value ===
      "string"
    &&
    value.length >
      MAX_STRING_LENGTH
  ) {
    throw new Error(
      "Une valeur texte dépasse la limite autorisée."
    );
  }
}


function validateRows(
  rows:
    ModelPredictionRow[]
): void {
  if (
    rows.length <
      1
    ||
    rows.length >
      MAX_ROWS
  ) {
    throw new Error(
      "La prédiction doit contenir entre 1 et 100 lignes."
    );
  }


  let cellCount =
    0;


  for (
    const row
    of rows
  ) {
    const entries =
      Object.entries(
        row
      );


    if (
      entries.length <
        1
    ) {
      throw new Error(
        "Une ligne de prédiction ne peut pas être vide."
      );
    }


    if (
      entries.length >
        MAX_COLUMNS
    ) {
      throw new Error(
        "Une ligne de prédiction contient trop de variables."
      );
    }


    for (
      const [
        rawKey,
        value,
      ]
      of entries
    ) {
      const key =
        rawKey.trim();


      if (
        !key
        ||
        key !==
          rawKey
      ) {
        throw new Error(
          "Les noms de variables de prédiction sont invalides."
        );
      }


      validateScalar(
        value
      );
    }


    cellCount +=
      entries.length;


    if (
      cellCount >
        MAX_CELLS
    ) {
      throw new Error(
        "La requête de prédiction contient trop de cellules."
      );
    }
  }
}


/* ============================================================
   RESPONSE
============================================================ */


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


function errorMessage(
  payload:
    unknown
): string {
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


    const detail =
      record.detail;


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
      typeof record.detail ===
        "string"
    ) {
      return (
        record.detail
      );
    }


    if (
      typeof record.message ===
        "string"
    ) {
      return (
        record.message
      );
    }
  }


  if (
    typeof payload ===
      "string"
  ) {
    return payload;
  }


  return (
    "La requête de prédiction a échoué."
  );
}


/* ============================================================
   PREDICT
============================================================ */


export async function predictModelLabRows(
  request:
    ModelPredictionRequest,

  signal?:
    AbortSignal
): Promise<
  ModelPredictionResponse
> {
  const workflowId =
    requiredIdentifier(
      request.workflow_id,
      "workflow_id"
    );


  const modelId =
    requiredIdentifier(
      request.model_id,
      "model_id"
    );


  validateRows(
    request.rows
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
              workflowId,

            model_id:
              modelId,

            rows:
              request.rows,
          }),

        signal,
      }
    );


  const payload =
    await readJsonResponse(
      response
    );


  if (
    !response.ok
  ) {
    throw new ModelLabApiError(
      errorMessage(
        payload
      ),

      response.status,

      payload
    );
  }


  return (
    payload as
      ModelPredictionResponse
  );
}
