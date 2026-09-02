import type {
  ModelLabEstimatorHyperparameters,
  ModelLabModelDetail,
  ModelLabPreprocessingContract,
  ModelLabProblemType,
  ModelLabSplitContract,
} from "./modelLabTypes";


export const MODEL_TRAINING_FRONTEND_CONTRACT_VERSION =
  "model_training_frontend_contract_v0.1" as const;


export const MODEL_TRAINING_API_CONTRACT_VERSION =
  "model_training_api_contract_v0.1" as const;


export const MODEL_TRAINING_API_VERSION =
  "model_training_api_v0.1" as const;


/* ============================================================
   TRAINING CONTEXT
============================================================ */


export type ModelTrainingColumnKind =
  | "numeric"
  | "boolean"
  | "datetime"
  | "categorical"
  | "other";


export type ModelTrainingAnalyticalType =
  | "unknown"
  | "identifier"
  | "categorical"
  | "temporal"
  | "quantitative"
  | "text";


export type ModelTrainingColumn = {
  name:
    string;

  kind:
    ModelTrainingColumnKind;

  nullable:
    boolean;

  analytical_type:
    ModelTrainingAnalyticalType;

  analytical_subtype:
    string
    | null;

  ml_eligible_as_target:
    boolean;

  ml_eligible_as_feature:
    boolean;

  exclusion_reason:
    string
    | null;

  rule_version:
    "model_training_api_contract_v0.1";
};


export type ModelTrainingDataset = {
  dataset_id:
    string;

  filename:
    string;

  row_count:
    number;

  column_count:
    number;

  columns:
    ModelTrainingColumn[];

  rule_version:
    "model_training_api_contract_v0.1";
};


export type ModelTrainingContextResponse = {
  workflow_id:
    string;

  preparation_session_revision:
    number;

  dataset_count:
    number;

  datasets:
    ModelTrainingDataset[];

  rule_version:
    "model_training_api_contract_v0.1";
};


/* ============================================================
   TRAINING CONTRACT
============================================================ */


export type ModelTrainingContract = {
  workflow_id:
    string;

  dataset_id:
    string;

  problem_type:
    ModelLabProblemType;

  target_column:
    string;

  feature_columns:
    string[];

  categorical_feature_columns:
    string[];

  estimator_key:
    string;

  estimator_hyperparameters?:
    ModelLabEstimatorHyperparameters
    | null;

  preprocessing:
    ModelLabPreprocessingContract;

  split:
    ModelLabSplitContract;

  rule_version?:
    "ml_training_contract_v0.1";
};


export type ModelTrainingRequest = {
  training:
    ModelTrainingContract;

  expected_preparation_session_revision:
    number;

  rule_version?:
    "model_training_request_v0.1";
};


/* ============================================================
   RESULT
============================================================ */


export type ModelTrainingResult =
  ModelLabModelDetail;


/* ============================================================
   API ERROR
============================================================ */


export type ModelTrainingApiErrorDetail = {
  error:
    string;

  message:
    string;

  workflow_id:
    string
    | null;

  retryable:
    boolean;

  api_version:
    "model_training_api_v0.1";
};
