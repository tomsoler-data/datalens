/* ============================================================
   MODEL LAB FRONTEND CONTRACTS
   MODEL_LAB_FRONTEND_CONTRACT_V0_1

   These are public HTTP / UI contracts only.

   Deliberately absent:
   - model_path
   - model bytes
   - model SHA
   - estimator object
   - complete Training Contract
============================================================ */


export const MODEL_LAB_FRONTEND_CONTRACT_VERSION =
  "model_lab_frontend_contract_v0.1" as const;


export const MODEL_LAB_API_CONTRACT_VERSION =
  "model_lab_api_contract_v0.1" as const;


export const MODEL_LAB_API_VERSION =
  "model_lab_api_v0.1" as const;


/* ============================================================
   JSON
============================================================ */


export type ModelLabJsonPrimitive =
  | string
  | number
  | boolean
  | null;


export type ModelLabJsonValue =
  | ModelLabJsonPrimitive
  | ModelLabJsonObject
  | ModelLabJsonValue[];


export type ModelLabJsonObject = {
  [key: string]:
    ModelLabJsonValue;
};


/* ============================================================
   SHARED TYPES
============================================================ */


export type ModelLabProblemType =
  | "regression"
  | "classification";


export type ModelLabPredictionFeatureValue =
  | string
  | number
  | boolean
  | null;


export type ModelLabPredictionScalar =
  | string
  | number
  | boolean;


/* ============================================================
   PREPROCESSING
============================================================ */


export type ModelLabPreprocessingContract = {
  numeric_imputation:
    | "error"
    | "median";

  categorical_imputation:
    | "error"
    | "most_frequent";

  categorical_encoding:
    "one_hot";

  handle_unknown_categories:
    "ignore";

  scale_numeric:
    boolean;

  rule_version:
    "ml_preprocessing_contract_v0.1";
};


/* ============================================================
   SPLIT
============================================================ */


export type ModelLabSplitContract = {
  strategy:
    "holdout";

  test_size:
    number;

  random_seed:
    number;

  shuffle:
    boolean;

  stratify:
    boolean;
};


/* ============================================================
   ESTIMATOR HYPERPARAMETERS
============================================================ */


type ModelLabEstimatorContractBase = {
  rule_version:
    "ml_estimator_contract_v0.1";
};


export type ModelLabLinearRegressionHyperparameters =
  ModelLabEstimatorContractBase & {
    kind:
      "linear_regression";

    fit_intercept:
      boolean;
  };


export type ModelLabRidgeRegressionHyperparameters =
  ModelLabEstimatorContractBase & {
    kind:
      "ridge_regression";

    alpha:
      number;

    fit_intercept:
      boolean;
  };


export type ModelLabLogisticRegressionHyperparameters =
  ModelLabEstimatorContractBase & {
    kind:
      "logistic_regression";

    inverse_regularization_strength:
      number;

    fit_intercept:
      boolean;

    max_iter:
      number;

    class_weight:
      | "balanced"
      | null;
  };


export type ModelLabRandomForestRegressorHyperparameters =
  ModelLabEstimatorContractBase & {
    kind:
      "random_forest_regressor";

    n_estimators:
      number;

    max_depth:
      number
      | null;

    min_samples_split:
      number;

    min_samples_leaf:
      number;

    max_features:
      | "sqrt"
      | "log2"
      | null;

    bootstrap:
      boolean;
  };


export type ModelLabRandomForestClassifierHyperparameters =
  ModelLabEstimatorContractBase & {
    kind:
      "random_forest_classifier";

    n_estimators:
      number;

    max_depth:
      number
      | null;

    min_samples_split:
      number;

    min_samples_leaf:
      number;

    max_features:
      | "sqrt"
      | "log2"
      | null;

    bootstrap:
      boolean;

    class_weight:
      | "balanced"
      | "balanced_subsample"
      | null;
  };


export type ModelLabEstimatorHyperparameters =
  | ModelLabLinearRegressionHyperparameters
  | ModelLabRidgeRegressionHyperparameters
  | ModelLabLogisticRegressionHyperparameters
  | ModelLabRandomForestRegressorHyperparameters
  | ModelLabRandomForestClassifierHyperparameters;


/* ============================================================
   MODEL CARD
============================================================ */


export type ModelLabModelCard = {
  model_id:
    string;

  workflow_id:
    string;

  dataset_id:
    string;

  problem_type:
    ModelLabProblemType;

  target_column:
    string;

  estimator_key:
    string;

  feature_columns:
    string[];

  categorical_feature_columns:
    string[];

  metrics:
    Record<
      string,
      number
    >;

  train_rows:
    number;

  test_rows:
    number;

  created_at_utc:
    string;

  experiment_id:
    string
    | null;

  preparation_session_revision:
    number
    | null;

  training_contract_sha256:
    string
    | null;

  has_experiment_provenance:
    boolean;

  rule_version:
    "model_lab_api_contract_v0.1";
};


/* ============================================================
   MODEL DETAIL
============================================================ */


export type ModelLabModelDetail =
  ModelLabModelCard & {
    preprocessing:
      ModelLabPreprocessingContract;

    split:
      ModelLabSplitContract;

    effective_estimator_hyperparameters:
      ModelLabEstimatorHyperparameters;
  };


/* ============================================================
   MODEL LIST
============================================================ */


export type ModelLabModelListResponse = {
  workflow_id:
    string;

  model_count:
    number;

  models:
    ModelLabModelCard[];

  ordering:
    "created_at_desc_model_id_asc";

  rule_version:
    "model_lab_api_contract_v0.1";
};


/* ============================================================
   DECISION THRESHOLD REQUEST
============================================================ */


export type ModelLabDecisionThresholdRequest = {
  threshold:
    number;
};


/* ============================================================
   EVALUATION ? PUBLIC CALL OPTIONS

   The UI may choose only the explicit threshold.

   Fixed evaluation policies remain server-owned and therefore
   are intentionally absent from this frontend input type.
============================================================ */


export type ModelLabEvaluationOptions = {
  decision_threshold?:
    ModelLabDecisionThresholdRequest;
};


/* ============================================================
   EVALUATION SUMMARY CONTRACT ? RESPONSE
============================================================ */


export type ModelLabEvaluationSummaryContract = {
  decision_threshold:
    (
      {
        threshold:
          number;

        method:
          "holdout_binary_decision_threshold";

        score_source:
          "predict_proba";

        positive_class_policy:
          "estimator_classes_index_1";

        comparison_operator:
          "greater_than_or_equal";

        threshold_selection_policy:
          "evaluate_requested_threshold_only";

        zero_division_policy:
          "zero";

        rule_version:
          "ml_decision_threshold_v0.1";
      }
      | null
    );

  method:
    "trusted_model_evaluation_summary";

  evaluation_scope:
    "persisted_model_holdout";

  evidence_policy:
    "server_reconstructed_only";

  selection_policy:
    "preserve_upstream_selection_only";

  explainability_policy:
    "default_permutation_importance_v0.1";

  threshold_policy:
    "explicit_requested_threshold_only";

  rule_version:
    "ml_model_evaluation_summary_v0.1";
};


/* ============================================================
   SELECTION EVIDENCE
============================================================ */


export type ModelLabSelectionEvidence = {
  source:
    | "standalone_model"
    | "model_comparison"
    | "tuned_model_promotion";

  status:
    | "selection_not_available"
    | "verified_selected";

  rank:
    number
    | null;

  selection_policy:
    | "regression_rmse_v0.1"
    | "classification_f1_macro_v0.1"
    | "rank_1_only"
    | null;

  primary_metric:
    | "rmse"
    | "f1_macro"
    | null;

  primary_metric_value:
    number
    | null;

  metric_scope:
    | "not_available"
    | "final_holdout"
    | "inner_cross_validation";

  rule_version:
    "ml_model_selection_evidence_v0.1";
};


/* ============================================================
   BASELINE
============================================================ */


export type ModelLabBaselineEvaluation = {
  problem_type:
    ModelLabProblemType;

  strategy:
    | "mean_train_target"
    | "majority_train_class";

  primary_metric:
    | "rmse"
    | "f1_macro";

  train_rows:
    number;

  test_rows:
    number;

  metrics:
    Record<
      string,
      number
    >;

  rule_version:
    "ml_baseline_v0.1";
};


export type ModelLabBaselineComparison = {
  problem_type:
    ModelLabProblemType;

  primary_metric:
    | "rmse"
    | "f1_macro";

  model_primary_metric_value:
    number;

  baseline_primary_metric_value:
    number;

  absolute_improvement:
    number;

  relative_improvement_pct:
    number
    | null;

  beats_baseline:
    boolean;

  rule_version:
    "ml_baseline_v0.1";
};


/* ============================================================
   CLASSIFICATION DIAGNOSTICS
============================================================ */


export type ModelLabClassificationMetricAverage = {
  precision:
    number;

  recall:
    number;

  f1:
    number;
};


export type ModelLabClassificationClassDiagnostics = {
  class_label:
    string;

  precision:
    number;

  recall:
    number;

  f1:
    number;

  support:
    number;

  true_positive:
    number;

  false_positive:
    number;

  false_negative:
    number;

  true_negative:
    number;
};


export type ModelLabClassificationDiagnostics = {
  workflow_id:
    string;

  dataset_id:
    string;

  model_id:
    string;

  experiment_id:
    string;

  problem_type:
    "classification";

  target_column:
    string;

  estimator_key:
    string;

  preparation_session_revision:
    number;

  training_contract_sha256:
    string;

  evaluation_rows:
    number;

  class_count:
    number;

  class_labels:
    string[];

  confusion_matrix:
    number[][];

  per_class:
    ModelLabClassificationClassDiagnostics[];

  accuracy:
    number;

  balanced_accuracy:
    number;

  macro_average:
    ModelLabClassificationMetricAverage;

  weighted_average:
    ModelLabClassificationMetricAverage;

  method:
    "holdout_classification_diagnostics";

  label_order_policy:
    "estimator_classes";

  zero_division_policy:
    "zero";

  rule_version:
    "ml_classification_diagnostics_v0.1";
};


/* ============================================================
   EXPLAINABILITY
============================================================ */


export type ModelLabFeatureImportance = {
  feature_name:
    string;

  rank:
    number;

  importance_mean:
    number;

  importance_std:
    number;
};


export type ModelLabExplainability = {
  workflow_id:
    string;

  dataset_id:
    string;

  model_id:
    string;

  experiment_id:
    string;

  problem_type:
    ModelLabProblemType;

  estimator_key:
    string;

  preparation_session_revision:
    number;

  training_contract_sha256:
    string;

  method:
    "permutation_importance";

  scoring:
    | "neg_root_mean_squared_error"
    | "f1_macro";

  n_repeats:
    number;

  random_seed:
    number;

  evaluation_rows:
    number;

  feature_importances:
    ModelLabFeatureImportance[];

  rule_version:
    "ml_model_explainability_v0.1";
};


/* ============================================================
   EVALUATION SUMMARY
============================================================ */


export type ModelLabEvaluationSummary = {
  workflow_id:
    string;

  dataset_id:
    string;

  model_id:
    string;

  experiment_id:
    string;

  problem_type:
    ModelLabProblemType;

  target_column:
    string;

  estimator_key:
    string;

  preparation_session_revision:
    number;

  training_contract_sha256:
    string;

  train_rows:
    number;

  test_rows:
    number;

  summary_contract:
    ModelLabEvaluationSummaryContract;

  metrics:
    Record<
      string,
      number
    >;

  baseline:
    ModelLabBaselineEvaluation;

  baseline_comparison:
    ModelLabBaselineComparison;

  selection_evidence:
    ModelLabSelectionEvidence;

  classification_diagnostics:
    ModelLabClassificationDiagnostics
    | null;

  decision_threshold_evaluation:
    ModelLabJsonObject
    | null;

  explainability:
    ModelLabExplainability;

  limitations:
    string[];

  evaluation_status:
    "complete";

  method:
    "trusted_model_evaluation_summary";

  rule_version:
    "ml_model_evaluation_summary_v0.1";
};


/* ============================================================
   PREDICTION
============================================================ */


export type ModelLabPredictionRow =
  Record<
    string,
    ModelLabPredictionFeatureValue
  >;


export type ModelLabPredictResponse = {
  workflow_id:
    string;

  model_id:
    string;

  problem_type:
    ModelLabProblemType;

  target_column:
    string;

  prediction_count:
    number;

  predictions:
    ModelLabPredictionScalar[];

  method:
    "trusted_native_predict";

  rule_version:
    "model_lab_api_contract_v0.1";
};


/* ============================================================
   STRUCTURED API ERROR
============================================================ */


export type ModelLabApiErrorDetail = {
  error:
    string;

  message:
    string;

  workflow_id:
    string
    | null;

  model_id:
    string
    | null;

  retryable:
    boolean;

  api_version:
    "model_lab_api_v0.1";
};
