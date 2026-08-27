export type TransformationOperation =
  | "derive_arithmetic"
  | "cast"
  | "bin_numeric"
  | "extract_date_part"
  | "aggregate";


export type TransformationStatus =
  | "validated"
  | "review_required";


export type TransformationRisk =
  | "low"
  | "medium"
  | "high";


export type TransformationArithmeticOperator =
  | "add"
  | "subtract"
  | "multiply"
  | "divide";


export type TransformationOperandKind =
  | "column"
  | "literal";


export type TransformationCastTargetType =
  | "string"
  | "integer"
  | "float"
  | "boolean"
  | "datetime";


export type TransformationDatePart =
  | "year"
  | "month"
  | "day"
  | "quarter"
  | "week"
  | "weekday";


export type TransformationAggregationFunction =
  | "sum"
  | "mean"
  | "median"
  | "min"
  | "max"
  | "count"
  | "nunique";


export type TransformationOperand = {
  kind:
    TransformationOperandKind;

  column?:
    string |
    null;

  value?:
    unknown;
};


type TransformationIntentBase = {
  request_id:
    string;

  dataset_id:
    string;

  /*
   * The backend reconciles the filename against the
   * server-owned artifact. Keeping it optional mirrors
   * the browser-facing API contract.
   */
  dataset_filename?:
    string;
};


export type DeriveArithmeticTransformationIntent =
  TransformationIntentBase & {
    operation:
      "derive_arithmetic";

    output_column:
      string;

    left:
      TransformationOperand;

    operator:
      TransformationArithmeticOperator;

    right:
      TransformationOperand;
  };


export type CastTransformationIntent =
  TransformationIntentBase & {
    operation:
      "cast";

    source_column:
      string;

    output_column:
      string;

    target_type:
      TransformationCastTargetType;
  };


export type BinNumericTransformationIntent =
  TransformationIntentBase & {
    operation:
      "bin_numeric";

    source_column:
      string;

    output_column:
      string;

    bins:
      number[];

    labels?:
      string[] |
      null;

    include_lowest?:
      boolean;

    right?:
      boolean;
  };


export type ExtractDatePartTransformationIntent =
  TransformationIntentBase & {
    operation:
      "extract_date_part";

    source_column:
      string;

    output_column:
      string;

    part:
      TransformationDatePart;
  };


export type TransformationAggregationMetric = {
  source_column:
    string;

  function:
    TransformationAggregationFunction;

  output_column:
    string;
};


export type AggregateTransformationIntent =
  TransformationIntentBase & {
    operation:
      "aggregate";

    group_by:
      string[];

    metrics:
      TransformationAggregationMetric[];

    output_dataset_id:
      string;

    output_dataset_filename:
      string;
  };


export type PreparationTransformationIntent =
  | DeriveArithmeticTransformationIntent
  | CastTransformationIntent
  | BinNumericTransformationIntent
  | ExtractDatePartTransformationIntent
  | AggregateTransformationIntent;


export type PreparationTransformationStep = {
  step_id:
    string;

  request_id:
    string;

  dataset_id:
    string;

  dataset_filename:
    string;

  operation:
    TransformationOperation;

  status:
    TransformationStatus;

  risk:
    TransformationRisk;

  input_columns:
    string[];

  output_column:
    string |
    null;

  output_dataset_id:
    string |
    null;

  output_dataset_filename:
    string |
    null;

  parameters:
    Record<
      string,
      unknown
    >;

  rationale:
    string;

  requires_human_approval:
    boolean;

  executable:
    boolean;
};


export type PreparationTransformationPlan = {
  dataset_id:
    string;

  dataset_filename:
    string;

  request_count:
    number;

  step_count:
    number;

  validated_count:
    number;

  review_required_count:
    number;

  human_approval_required_count:
    number;

  ready_for_approval:
    boolean;

  steps:
    PreparationTransformationStep[];

  notes:
    string[];

  rule_version:
    string;
};


export type TransformationApprovalDecision =
  | "approve"
  | "reject"
  | "defer";


export type PreparationTransformationApprovalCommand = {
  request_id:
    string;

  decision:
    TransformationApprovalDecision;

  actor?:
    string;

  comment?:
    string |
    null;

  decided_at?:
    string |
    null;
};


export type PreparationTransformationMaterialization = {
  workflow_id:
    string;

  source_dataset_id:
    string;

  persisted_dataset_ids:
    string[];

  derived_dataset_ids:
    string[];

  artifact_count:
    number;

  source_data_changed:
    boolean;

  materialization_kind:
    string;

  bridge_version:
    string;
};


export type PreparationTransformationApplyStatus =
  | "ready"
  | "skipped"
  | "validation_failed"
  | string;


/*
 * The UI currently needs the plan and the materialization
 * contract in detail.
 *
 * Approval, execution and independent validation remain
 * server-owned reports. Their detailed schemas can be made
 * stricter later when the UI needs individual checks.
 */
export type PreparationTransformationApplyResponse = {
  status:
    PreparationTransformationApplyStatus;

  plan:
    PreparationTransformationPlan;

  approved_plan:
    Record<
      string,
      unknown
    > |
    null;

  execution:
    Record<
      string,
      unknown
    > |
    null;

  validation:
    Record<
      string,
      unknown
    > |
    null;

  materialization:
    PreparationTransformationMaterialization |
    null;

  notes:
    string[];

  api_version:
    string;
};