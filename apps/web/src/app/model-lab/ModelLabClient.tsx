"use client";

// DATALENS_MODEL_LAB_UNIFIED_SHELL_V0_1

// DATALENS_MODEL_LAB_PRODUCT_SHELL_V0_1
// DATALENS_MODEL_LAB_TRAINING_WORKFLOW_V0_1

import Link
  from "next/link";

import {
  useSearchParams,
  useRouter
} from "next/navigation";

import {
  useEffect,
  useMemo,
  useState,
  useSyncExternalStore,
} from "react";

import {
  evaluateModelLabModel,
  getModelLabModelDetail,
  listModelLabModels,
  ModelLabApiError,
} from "../../components/modelLab/modelLabApi";

import {
  getModelTrainingContext,
  trainModel,
} from "../../components/modelLab/modelTrainingApi";

import {
  predictModelLabRows,
} from "../../components/modelLab/modelPredictionApi";


import ModelObservabilityPanel
  from "../../components/modelLab/ModelObservabilityPanel";

import type {
  ModelLabEstimatorHyperparameters,
  ModelLabModelCard,
  ModelLabEvaluationSummary,
  ModelLabModelDetail,
  ModelLabModelListResponse,
  ModelLabProblemType,
} from "../../components/modelLab/modelLabTypes";

import type {
  ModelTrainingContextResponse,
  ModelTrainingDataset,
  ModelTrainingRequest,
} from "../../components/modelLab/modelTrainingTypes";

import type {
  ModelPredictionResponse,
  ModelPredictionRow,
  ModelPredictionScalar,
} from "../../components/modelLab/modelPredictionApi";

import {
  readActivePreparationWorkflowId,
} from "../../components/preparation/preparationWorkflowStorage";

import WorkspaceNavigation
  from "../../components/workspace/WorkspaceNavigation";

import type {
  WorkspaceStep,
} from "../../components/workspace/workspaceNavigationTypes";

import {
  persistActiveWorkspaceStep,
} from "../../components/workspace/workspaceNavigationStorage";

import workspaceStyles
  from "../page.module.css";

import styles
  from "./modelLab.module.css";


type WorkflowResolutionSource =
  | "query"
  | "browser_storage"
  | "none";


type ModelLabConnectionState =
  | "idle"
  | "loading"
  | "ready"
  | "error";


type ModelLabInventoryState = {
  workflowId:
    string;

  status:
    "ready"
    | "error";

  inventory:
    ModelLabModelListResponse | null;

  error:
    string | null;
};


type ModelLabDetailState = {
  workflowId:
    string;

  modelId:
    string;

  status:
    "ready"
    | "error";

  detail:
    ModelLabModelDetail | null;

  error:
    string | null;
};


type ModelLabPredictionState = {
  workflowId:
    string;

  modelId:
    string;

  status:
    "ready"
    | "error";

  result:
    ModelPredictionResponse | null;

  error:
    string | null;
};


type ModelLabPredictionFormState = {
  modelId:
    string;

  values:
    Record<
      string,
      string
    >;
};


type ModelLabEvaluationState = {
  workflowId:
    string;

  modelId:
    string;

  status:
    "ready"
    | "error";

  evaluation:
    ModelLabEvaluationSummary | null;

  error:
    string | null;
};


type ModelTrainingContextState = {
  workflowId:
    string;

  status:
    "ready"
    | "error";

  context:
    ModelTrainingContextResponse | null;

  error:
    string | null;
};


type ModelTrainingSplitStrategy =
  | ""
  | "holdout"
  | "group_holdout"
  | "time_holdout"
  | "purged_group_time_holdout";


/* ============================================================
   ACTIVE WORKFLOW STORAGE
============================================================ */


function subscribeToActiveWorkflow(
  onStoreChange:
    () => void
): () => void {
  if (
    typeof window ===
      "undefined"
  ) {
    return (
      () => undefined
    );
  }


  function handleStorage(
    event:
      StorageEvent
  ) {
    if (
      event.storageArea ===
        window.localStorage
    ) {
      onStoreChange();
    }
  }


  window.addEventListener(
    "storage",
    handleStorage
  );


  return () => {
    window.removeEventListener(
      "storage",
      handleStorage
    );
  };
}


function activeWorkflowSnapshot():
  string | null {
  return (
    readActivePreparationWorkflowId()
  );
}


function activeWorkflowServerSnapshot():
  string | null {
  return null;
}


/* ============================================================
   PRESENTATION
============================================================ */


function workflowSourceLabel(
  source:
    WorkflowResolutionSource
): string {
  if (
    source ===
      "query"
  ) {
    return (
      "URL explicite"
    );
  }


  if (
    source ===
      "browser_storage"
  ) {
    return (
      "Workflow actif"
    );
  }


  return (
    "Non résolu"
  );
}


function connectionLabel(
  state:
    ModelLabConnectionState
): string {
  if (
    state ===
      "loading"
  ) {
    return (
      "Connexion…"
    );
  }


  if (
    state ===
      "ready"
  ) {
    return (
      "Disponible"
    );
  }


  if (
    state ===
      "error"
  ) {
    return (
      "Erreur"
    );
  }


  return (
    "En attente"
  );
}


function problemTypeLabel(
  problemType:
    ModelLabProblemType
): string {
  return (
    problemType ===
      "classification"
      ? "Classification"
      : "Régression"
  );
}


function estimatorLabel(
  estimatorKey:
    string
): string {
  const labels:
    Record<
      string,
      string
    > = {
      linear_regression:
        "Régression linéaire",

      ridge_regression:
        "Régression Ridge",

      logistic_regression:
        "Régression logistique",

      random_forest_regressor:
        "Random Forest Regressor",

      random_forest_classifier:
        "Random Forest Classifier",
    };


  return (
    labels[
      estimatorKey
    ] ??
    estimatorKey
  );
}


function metricLabel(
  metricName:
    string
): string {
  const labels:
    Record<
      string,
      string
    > = {
      accuracy:
        "Accuracy",

      precision_macro:
        "Precision macro",

      recall_macro:
        "Recall macro",

      f1_macro:
        "F1 macro",

      roc_auc:
        "ROC AUC",

      mae:
        "MAE",

      mse:
        "MSE",

      rmse:
        "RMSE",

      r2:
        "R²",
    };


  return (
    labels[
      metricName
    ] ??
    metricName
  );
}


function formatMetric(
  value:
    number
): string {
  if (
    Math.abs(
      value
    ) >=
      1000
  ) {
    return (
      new Intl.NumberFormat(
        "fr-FR",
        {
          maximumFractionDigits:
            2,
        }
      ).format(
        value
      )
    );
  }


  return (
    new Intl.NumberFormat(
      "fr-FR",
      {
        maximumFractionDigits:
          4,
      }
    ).format(
      value
    )
  );
}


function formatDate(
  value:
    string
): string {
  const date =
    new Date(
      value
    );


  if (
    Number.isNaN(
      date.getTime()
    )
  ) {
    return value;
  }


  return (
    new Intl.DateTimeFormat(
      "fr-FR",
      {
        dateStyle:
          "medium",

        timeStyle:
          "short",
      }
    ).format(
      date
    )
  );
}


function shortIdentifier(
  value:
    string
): string {
  if (
    value.length <=
      24
  ) {
    return value;
  }


  return (
    `${value.slice(0, 12)}…${value.slice(-8)}`
  );
}


function errorMessage(
  error:
    unknown
): string {
  if (
    error instanceof
      ModelLabApiError
  ) {
    return (
      error.message
    );
  }


  if (
    error instanceof
      Error
  ) {
    return (
      error.message
    );
  }


  return (
    "Impossible de charger le Model Lab."
  );
}


function primaryMetricEntry(
  model:
    ModelLabModelCard
): [
  string,
  number
] | null {
  const preferredNames =
    model.problem_type ===
      "classification"
      ? [
          "f1_macro",
          "accuracy",
          "roc_auc",
        ]
      : [
          "rmse",
          "mae",
          "r2",
        ];


  for (
    const name
    of preferredNames
  ) {
    const value =
      model.metrics[
        name
      ];


    if (
      typeof value ===
        "number"
    ) {
      return [
        name,
        value,
      ];
    }
  }


  const first =
    Object.entries(
      model.metrics
    )[0];


  return (
    first ??
    null
  );
}


function predictionDisplayValue(
  value:
    ModelPredictionScalar
): string {
  if (
    value ===
      null
  ) {
    return (
      "Valeur manquante"
    );
  }


  if (
    typeof value ===
      "boolean"
  ) {
    return (
      value
        ? "Oui"
        : "Non"
    );
  }


  if (
    typeof value ===
      "number"
  ) {
    return (
      formatMetric(
        value
      )
    );
  }


  return String(
    value
  );
}


function evaluationLimitationLabel(
  value:
    string
): string {
  const labels:
    Record<
      string,
      string
    > = {
      single_holdout_evaluation:
        "évaluation sur un seul holdout",

      no_external_validation:
        "Aucune validation externe",

      feature_importance_not_causal:
        "L'importance des variables n'est pas causale",

      selection_evidence_not_available:
        "Preuve de sélection non disponible",

      decision_threshold_not_included:
        "Aucun seuil de décision spécifique",

      requested_threshold_not_optimized:
        "Le seuil demandé n'est pas optimisé",
    };


  return (
    labels[
      value
    ] ??
    value
  );
}


function hyperparameterEntries(
  value:
    ModelLabEstimatorHyperparameters
): [
  string,
  string
][] {
  return (
    Object.entries(
      value
    )
      .filter(
        (
          [
            key,
          ]
        ) =>
          key !==
          "rule_version"
      )
      .map(
        (
          [
            key,
            rawValue,
          ]
        ) => [
          key,
          rawValue ===
            null
            ? "Aucun"
            : String(
                rawValue
              ),
        ]
      )
  );
}


function estimatorOptions(
  problemType:
    ModelLabProblemType
): Array<{
  value:
    string;

  label:
    string;
}> {
  if (
    problemType ===
      "classification"
  ) {
    return [
      {
        value:
          "logistic_regression",

        label:
          "Régression logistique",
      },

      {
        value:
          "random_forest_classifier",

        label:
          "Random Forest Classifier",
      },
    ];
  }


  return [
    {
      value:
        "linear_regression",

      label:
        "Régression linéaire",
    },

    {
      value:
        "ridge_regression",

      label:
        "Régression Ridge",
    },

    {
      value:
        "random_forest_regressor",

      label:
        "Random Forest Regressor",
    },
  ];
}


function trainingColumnKindLabel(
  kind:
    string
): string {
  const labels:
    Record<
      string,
      string
    > = {
      numeric:
        "Numérique",

      categorical:
        "Catégorielle",

      boolean:
        "Booléenne",

      datetime:
        "Date / heure",

      other:
        "Autre",
    };


  return (
    labels[
      kind
    ] ??
    kind
  );
}


function isDirectMLFeatureKind(
  kind:
    string
): boolean {
  return (
    kind ===
      "numeric"
    ||
    kind ===
      "categorical"
    ||
    kind ===
      "boolean"
  );
}


/* ============================================================
   CLIENT
============================================================ */


export default function ModelLabClient() {
  const router = useRouter();

  const searchParams =
    useSearchParams();


  const queryWorkflowId =
    (
      searchParams.get(
        "workflow_id"
      ) ??
      ""
    ).trim();


  const storedWorkflowId =
    useSyncExternalStore(
      subscribeToActiveWorkflow,
      activeWorkflowSnapshot,
      activeWorkflowServerSnapshot
    );


  const workflowId =
    queryWorkflowId ||
    storedWorkflowId;


  const workflowSource:
    WorkflowResolutionSource =
      queryWorkflowId
        ? "query"
        : (
            storedWorkflowId
              ? "browser_storage"
              : "none"
          );


  const [
    inventoryState,
    setInventoryState,
  ] = useState<
    ModelLabInventoryState | null
  >(
    null
  );


  const [
    selectedModelId,
    setSelectedModelId,
  ] = useState<
    string | null
  >(
    null
  );


  const [
    detailState,
    setDetailState,
  ] = useState<
    ModelLabDetailState | null
  >(
    null
  );


  const [
    inventoryRefreshKey,
    setInventoryRefreshKey,
  ] = useState(
    0
  );


  const [
    trainingOpen,
    setTrainingOpen,
  ] = useState(
    false
  );


  const [
    trainingContextState,
    setTrainingContextState,
  ] = useState<
    ModelTrainingContextState | null
  >(
    null
  );


  const [
    trainingDatasetId,
    setTrainingDatasetId,
  ] = useState<
    string | null
  >(
    null
  );


  const [
    trainingProblemType,
    setTrainingProblemType,
  ] = useState<
    ModelLabProblemType
  >(
    "classification"
  );


  const [
    trainingTargetColumn,
    setTrainingTargetColumn,
  ] = useState(
    ""
  );


  const [
    trainingEstimatorKey,
    setTrainingEstimatorKey,
  ] = useState(
    "logistic_regression"
  );


  const [
    trainingFeatureColumns,
    setTrainingFeatureColumns,
  ] = useState<
    string[]
  >(
    []
  );


  const [
    trainingSplitStrategy,
    setTrainingSplitStrategy,
  ] = useState<
    ModelTrainingSplitStrategy
  >(
    ""
  );


  const [
    trainingGroupColumn,
    setTrainingGroupColumn,
  ] = useState(
    ""
  );


  const [
    trainingTimeColumn,
    setTrainingTimeColumn,
  ] = useState(
    ""
  );


  const [
    predictionPanelModelId,
    setPredictionPanelModelId,
  ] = useState<
    string | null
  >(
    null
  );


  const [
    predictionFormState,
    setPredictionFormState,
  ] = useState<
    ModelLabPredictionFormState | null
  >(
    null
  );


  const [
    predictionState,
    setPredictionState,
  ] = useState<
    ModelLabPredictionState | null
  >(
    null
  );


  const [
    predictionSubmitting,
    setPredictionSubmitting,
  ] = useState(
    false
  );


  const [
    evaluationState,
    setEvaluationState,
  ] = useState<
    ModelLabEvaluationState | null
  >(
    null
  );


  const [
    evaluationSubmitting,
    setEvaluationSubmitting,
  ] = useState(
    false
  );


  const [
    trainingSubmitting,
    setTrainingSubmitting,
  ] = useState(
    false
  );


  const [
    trainingSubmitError,
    setTrainingSubmitError,
  ] = useState<
    string | null
  >(
    null
  );


  /* ========================================================
     INVENTORY
  ======================================================== */


  useEffect(
    () => {
      if (
        !workflowId
      ) {
        return;
      }


      const controller =
        new AbortController();


      void (
        async () => {
          try {
            const result =
              await listModelLabModels(
                workflowId,
                controller.signal
              );


            if (
              controller
                .signal
                .aborted
            ) {
              return;
            }


            setInventoryState({
              workflowId,

              status:
                "ready",

              inventory:
                result,

              error:
                null,
            });
          } catch (
            error
          ) {
            if (
              controller
                .signal
                .aborted
            ) {
              return;
            }


            setInventoryState({
              workflowId,

              status:
                "error",

              inventory:
                null,

              error:
                errorMessage(
                  error
                ),
            });
          }
        }
      )();


      return () => {
        controller.abort();
      };
    },
    [
      workflowId,
      inventoryRefreshKey,
    ]
  );


  const inventoryStateMatches =
    Boolean(
      workflowId &&
      inventoryState
        ?.workflowId ===
        workflowId
    );


  const inventory =
    inventoryStateMatches
      ? inventoryState
          ?.inventory ??
        null
      : null;


  const loadError =
    inventoryStateMatches
      ? inventoryState
          ?.error ??
        null
      : null;


  const connectionState:
    ModelLabConnectionState =
      !workflowId
        ? "idle"
        : (
            !inventoryStateMatches
              ? "loading"
              : (
                  inventoryState
                    ?.status ??
                  "loading"
                )
          );


  /* ========================================================
     MODEL TRAINING CONTEXT
  ======================================================== */


  useEffect(
    () => {
      if (
        !trainingOpen ||
        !workflowId
      ) {
        return;
      }


      const controller =
        new AbortController();


      void (
        async () => {
          try {
            const context =
              await getModelTrainingContext(
                workflowId,
                controller.signal
              );


            if (
              controller
                .signal
                .aborted
            ) {
              return;
            }


            setTrainingContextState({
              workflowId,

              status:
                "ready",

              context,

              error:
                null,
            });
          } catch (
            error
          ) {
            if (
              controller
                .signal
                .aborted
            ) {
              return;
            }


            setTrainingContextState({
              workflowId,

              status:
                "error",

              context:
                null,

              error:
                errorMessage(
                  error
                ),
            });
          }
        }
      )();


      return () => {
        controller.abort();
      };
    },
    [
      workflowId,
      trainingOpen,
    ]
  );


  const trainingContextMatches =
    Boolean(
      workflowId &&
      trainingContextState
        ?.workflowId ===
        workflowId
    );


  const trainingContext =
    trainingContextMatches &&
    trainingContextState
      ?.status ===
      "ready"
      ? trainingContextState
          .context
      : null;


  const trainingContextError =
    trainingContextMatches &&
    trainingContextState
      ?.status ===
      "error"
      ? trainingContextState
          .error
      : null;


  const trainingContextLoading =
    Boolean(
      trainingOpen &&
      workflowId &&
      !trainingContextMatches
    );


  const selectedTrainingDataset:
    ModelTrainingDataset | null =
      useMemo(
        () => {
          if (
            !trainingContext ||
            trainingContext
              .datasets
              .length ===
              0
          ) {
            return null;
          }


          if (
            trainingDatasetId
          ) {
            const explicit =
              trainingContext
                .datasets
                .find(
                  (
                    dataset
                  ) =>
                    dataset.dataset_id ===
                    trainingDatasetId
                );


            if (
              explicit
            ) {
              return explicit;
            }
          }


          return (
            trainingContext
              .datasets[
                0
              ]
          );
        },
        [
          trainingContext,
          trainingDatasetId,
        ]
      );


  const eligibleTargetColumns =
    useMemo(
      () => {
        if (
          !selectedTrainingDataset
        ) {
          return [];
        }


        return (
          selectedTrainingDataset
            .columns
            .filter(
              (
                column
              ) => {
                if (
                  !column.ml_eligible_as_target
                ) {
                  return false;
                }


                if (
                  trainingProblemType ===
                    "regression"
                ) {
                  return (
                    column.kind ===
                    "numeric"
                  );
                }


                return (
                  column.kind ===
                    "numeric"
                  ||
                  column.kind ===
                    "categorical"
                  ||
                  column.kind ===
                    "boolean"
                );
              }
            )
        );
      },
      [
        selectedTrainingDataset,
        trainingProblemType,
      ]
    );


  const eligibleFeatureColumns =
    useMemo(
      () => {
        if (
          !selectedTrainingDataset
        ) {
          return [];
        }


        return (
          selectedTrainingDataset
            .columns
            .filter(
              (
                column
              ) =>
                column.name !==
                  trainingTargetColumn
                &&
                column.ml_eligible_as_feature
                &&
                isDirectMLFeatureKind(
                  column.kind
                )
            )
        );
      },
      [
        selectedTrainingDataset,
        trainingTargetColumn,
      ]
    );


  const identifierTrainingColumns =
    useMemo(
      () => {
        if (
          !selectedTrainingDataset
        ) {
          return [];
        }


        return (
          selectedTrainingDataset
            .columns
            .filter(
              (
                column
              ) =>
                column.analytical_type ===
                  "identifier"
            )
        );
      },
      [
        selectedTrainingDataset,
      ]
    );


  const eligibleTrainingGroupColumns =
    useMemo(
      () => {
        if (
          !selectedTrainingDataset
        ) {
          return [];
        }


        return (
          selectedTrainingDataset
            .columns
            .filter(
              (
                column
              ) =>
                column.ml_eligible_as_group
            )
        );
      },
      [
        selectedTrainingDataset,
      ]
    );


  const eligibleTrainingTimeColumns =
    useMemo(
      () => {
        if (
          !selectedTrainingDataset
        ) {
          return [];
        }


        return (
          selectedTrainingDataset
            .columns
            .filter(
              (
                column
              ) =>
                column.ml_eligible_as_time
            )
        );
      },
      [
        selectedTrainingDataset,
      ]
    );


  const trainingSplitUsesGroup =
    (
      trainingSplitStrategy ===
        "group_holdout"
      ||
      trainingSplitStrategy ===
        "purged_group_time_holdout"
    );


  const trainingSplitUsesTime =
    (
      trainingSplitStrategy ===
        "time_holdout"
      ||
      trainingSplitStrategy ===
        "purged_group_time_holdout"
    );


  const trainingSplitIsPurgedGroupTime =
    (
      trainingSplitStrategy ===
        "purged_group_time_holdout"
    );


  const selectedTrainingFeatureSet =
    useMemo(
      () =>
        new Set(
          trainingFeatureColumns
        ),
      [
        trainingFeatureColumns,
      ]
    );


  const selectedCategoricalFeatures =
    useMemo(
      () => {
        if (
          !selectedTrainingDataset
        ) {
          return [];
        }


        const selected =
          new Set(
            trainingFeatureColumns
          );


        return (
          selectedTrainingDataset
            .columns
            .filter(
              (
                column
              ) =>
                selected.has(
                  column.name
                )
                &&
                (
                  column.kind ===
                    "categorical"
                  ||
                  column.kind ===
                    "boolean"
                )
            )
            .map(
              (
                column
              ) =>
                column.name
            )
        );
      },
      [
        selectedTrainingDataset,
        trainingFeatureColumns,
      ]
    );


  const selectedNumericFeatures =
    useMemo(
      () => {
        if (
          !selectedTrainingDataset
        ) {
          return [];
        }


        const selected =
          new Set(
            trainingFeatureColumns
          );


        return (
          selectedTrainingDataset
            .columns
            .filter(
              (
                column
              ) =>
                selected.has(
                  column.name
                )
                &&
                column.kind ===
                  "numeric"
            )
        );
      },
      [
        selectedTrainingDataset,
        trainingFeatureColumns,
      ]
    );


  const selectedCategoricalFeatureModels =
    useMemo(
      () => {
        if (
          !selectedTrainingDataset
        ) {
          return [];
        }


        const selected =
          new Set(
            selectedCategoricalFeatures
          );


        return (
          selectedTrainingDataset
            .columns
            .filter(
              (
                column
              ) =>
                selected.has(
                  column.name
                )
            )
        );
      },
      [
        selectedTrainingDataset,
        selectedCategoricalFeatures,
      ]
    );


  const trainingCanSubmit =
    Boolean(
      workflowId &&
      trainingContext &&
      selectedTrainingDataset &&
      trainingTargetColumn &&
      trainingFeatureColumns
        .length >
        0 &&
      Boolean(
        trainingSplitStrategy
      ) &&
      (
        !trainingSplitUsesGroup
        ||
        Boolean(
          trainingGroupColumn
        )
      ) &&
      (
        !trainingSplitUsesTime
        ||
        Boolean(
          trainingTimeColumn
        )
      ) &&
      !trainingSubmitting
    );


  function openTrainingPanel() {
    setTrainingContextState(
      null
    );

    setTrainingDatasetId(
      null
    );

    setTrainingProblemType(
      "classification"
    );

    setTrainingTargetColumn(
      ""
    );

    setTrainingEstimatorKey(
      "logistic_regression"
    );

    setTrainingFeatureColumns(
      []
    );

    setTrainingSplitStrategy(
      ""
    );

    setTrainingGroupColumn(
      ""
    );

    setTrainingTimeColumn(
      ""
    );

    setTrainingSubmitError(
      null
    );

    setTrainingOpen(
      true
    );
  }


  function closeTrainingPanel() {
    if (
      trainingSubmitting
    ) {
      return;
    }


    setTrainingOpen(
      false
    );

    setTrainingSubmitError(
      null
    );
  }


  function changeTrainingDataset(
    datasetId:
      string
  ) {
    setTrainingDatasetId(
      datasetId
    );

    setTrainingTargetColumn(
      ""
    );

    setTrainingFeatureColumns(
      []
    );

    setTrainingSplitStrategy(
      ""
    );

    setTrainingGroupColumn(
      ""
    );

    setTrainingTimeColumn(
      ""
    );

    setTrainingSubmitError(
      null
    );
  }


  function changeTrainingProblemType(
    problemType:
      ModelLabProblemType
  ) {
    setTrainingProblemType(
      problemType
    );


    setTrainingEstimatorKey(
      problemType ===
        "classification"
        ? "logistic_regression"
        : "linear_regression"
    );


    if (
      problemType ===
        "regression"
      &&
      selectedTrainingDataset
    ) {
      const target =
        selectedTrainingDataset
          .columns
          .find(
            (
              column
            ) =>
              column.name ===
              trainingTargetColumn
          );


      if (
        target &&
        target.kind !==
          "numeric"
      ) {
        setTrainingTargetColumn(
          ""
        );
      }
    }


    setTrainingSubmitError(
      null
    );
  }


  function changeTrainingTarget(
    targetColumn:
      string
  ) {
    setTrainingTargetColumn(
      targetColumn
    );


    setTrainingFeatureColumns(
      (
        current
      ) =>
        current.filter(
          (
            column
          ) =>
            column !==
            targetColumn
        )
    );


    setTrainingSubmitError(
      null
    );
  }


  function changeTrainingSplitStrategy(
    strategy:
      ModelTrainingSplitStrategy
  ) {
    setTrainingSplitStrategy(
      strategy
    );


    if (
      strategy !==
        "group_holdout"
      &&
      strategy !==
        "purged_group_time_holdout"
    ) {
      setTrainingGroupColumn(
        ""
      );
    }


    if (
      strategy !==
        "time_holdout"
      &&
      strategy !==
        "purged_group_time_holdout"
    ) {
      setTrainingTimeColumn(
        ""
      );
    }


    setTrainingSubmitError(
      null
    );
  }


  function toggleTrainingFeature(
    columnName:
      string
  ) {
    setTrainingFeatureColumns(
      (
        current
      ) => {
        if (
          current.includes(
            columnName
          )
        ) {
          return (
            current.filter(
              (
                column
              ) =>
                column !==
                columnName
            )
          );
        }


        return [
          ...current,
          columnName,
        ];
      }
    );


    setTrainingSubmitError(
      null
    );
  }


  async function submitTraining() {
    if (
      !workflowId ||
      !trainingContext ||
      !selectedTrainingDataset ||
      !trainingTargetColumn ||
      trainingFeatureColumns
        .length ===
        0 ||
      !trainingSplitStrategy ||
      (
        trainingSplitUsesGroup
        &&
        !trainingGroupColumn
      ) ||
      (
        trainingSplitUsesTime
        &&
        !trainingTimeColumn
      )
    ) {
      setTrainingSubmitError(
        "Complétez le contrat d’entraînement avant de lancer le modèle."
      );

      return;
    }


    const numericNeedsImputation =
      selectedNumericFeatures.some(
        (
          column
        ) =>
          column.nullable
      );


    const categoricalNeedsImputation =
      selectedCategoricalFeatureModels.some(
        (
          column
        ) =>
          column.nullable
      );


    const request:
      ModelTrainingRequest = {
        training:
          {
            workflow_id:
              workflowId,

            dataset_id:
              selectedTrainingDataset
                .dataset_id,

            problem_type:
              trainingProblemType,

            target_column:
              trainingTargetColumn,

            feature_columns:
              [
                ...trainingFeatureColumns,
              ],

            categorical_feature_columns:
              [
                ...selectedCategoricalFeatures,
              ],

            estimator_key:
              trainingEstimatorKey,

            preprocessing:
              {
                numeric_imputation:
                  numericNeedsImputation
                    ? "median"
                    : "error",

                categorical_imputation:
                  categoricalNeedsImputation
                    ? "most_frequent"
                    : "error",

                categorical_encoding:
                  "one_hot",

                handle_unknown_categories:
                  "ignore",

                scale_numeric:
                  true,

                rule_version:
                  "ml_preprocessing_contract_v0.1",
              },

            split:
              trainingSplitStrategy ===
                "purged_group_time_holdout"
                ? {
                    strategy:
                      "purged_group_time_holdout",

                    group_column:
                      trainingGroupColumn,

                    time_column:
                      trainingTimeColumn,

                    test_size:
                      0.2,

                    random_seed:
                      42,

                    shuffle:
                      false,

                    stratify:
                      false,
                  }
                : (
                    trainingSplitStrategy ===
                      "group_holdout"
                      ? {
                          strategy:
                            "group_holdout",

                          group_column:
                            trainingGroupColumn,

                          test_size:
                            0.2,

                          random_seed:
                            42,

                          shuffle:
                            true,

                          stratify:
                            false,
                        }
                      : (
                          trainingSplitStrategy ===
                            "time_holdout"
                            ? {
                                strategy:
                                  "time_holdout",

                                time_column:
                                  trainingTimeColumn,

                                test_size:
                                  0.2,

                                random_seed:
                                  42,

                                shuffle:
                                  false,

                                stratify:
                                  false,
                              }
                            : {
                                strategy:
                                  "holdout",

                                test_size:
                                  0.2,

                                random_seed:
                                  42,

                                shuffle:
                                  true,

                                stratify:
                                  trainingProblemType ===
                                    "classification",
                              }
                        )
                  ),
          },

        expected_preparation_session_revision:
          trainingContext
            .preparation_session_revision,
      };


    setTrainingSubmitting(
      true
    );

    setTrainingSubmitError(
      null
    );


    try {
      const result =
        await trainModel(
          request
        );


      setSelectedModelId(
        result.model_id
      );


      setDetailState({
        workflowId,

        modelId:
          result.model_id,

        status:
          "ready",

        detail:
          result,

        error:
          null,
      });


      setInventoryRefreshKey(
        (
          current
        ) =>
          current +
          1
      );


      setTrainingOpen(
        false
      );
    } catch (
      error
    ) {
      setTrainingSubmitError(
        errorMessage(
          error
        )
      );
    } finally {
      setTrainingSubmitting(
        false
      );
    }
  }


  /* ========================================================
     SELECTED MODEL

     When there is no explicit selection, the first model from
     the server-owned deterministic inventory becomes the
     visible default.

     No effect is needed to synchronize this derived value.
  ======================================================== */


  const selectedModel =
    useMemo(
      () => {
        if (
          !inventory ||
          inventory.models.length ===
            0
        ) {
          return null;
        }


        if (
          selectedModelId
        ) {
          const explicit =
            inventory.models.find(
              (
                model
              ) =>
                model.model_id ===
                selectedModelId
            );


          if (
            explicit
          ) {
            return explicit;
          }
        }


        return (
          inventory.models[
            0
          ]
        );
      },
      [
        inventory,
        selectedModelId,
      ]
    );


  const effectiveModelId =
    selectedModel
      ?.model_id ??
    null;


  /* ========================================================
     MODEL DETAIL
  ======================================================== */


  useEffect(
    () => {
      if (
        !workflowId ||
        !effectiveModelId
      ) {
        return;
      }


      const controller =
        new AbortController();


      void (
        async () => {
          try {
            const detail =
              await getModelLabModelDetail(
                workflowId,
                effectiveModelId,
                controller.signal
              );


            if (
              controller
                .signal
                .aborted
            ) {
              return;
            }


            setDetailState({
              workflowId,

              modelId:
                effectiveModelId,

              status:
                "ready",

              detail,

              error:
                null,
            });
          } catch (
            error
          ) {
            if (
              controller
                .signal
                .aborted
            ) {
              return;
            }


            setDetailState({
              workflowId,

              modelId:
                effectiveModelId,

              status:
                "error",

              detail:
                null,

              error:
                errorMessage(
                  error
                ),
            });
          }
        }
      )();


      return () => {
        controller.abort();
      };
    },
    [
      workflowId,
      effectiveModelId,
    ]
  );


  const detailStateMatches =
    Boolean(
      workflowId &&
      effectiveModelId &&
      detailState
        ?.workflowId ===
        workflowId &&
      detailState
        ?.modelId ===
        effectiveModelId
    );


  const selectedDetail =
    detailStateMatches &&
    detailState
      ?.status ===
      "ready"
      ? detailState
          .detail
      : null;


  const detailError =
    detailStateMatches &&
    detailState
      ?.status ===
      "error"
      ? detailState
          .error
      : null;


  const detailLoading =
    Boolean(
      workflowId &&
      effectiveModelId &&
      !detailStateMatches
    );


  /* ========================================================
     MODEL PREDICTION
  ======================================================== */


  const predictionPanelOpen =
    Boolean(
      effectiveModelId &&
      predictionPanelModelId ===
        effectiveModelId
    );


  const predictionFormValues =
    (
      effectiveModelId &&
      predictionFormState
        ?.modelId ===
        effectiveModelId
    )
      ? predictionFormState
          .values
      : {};


  const predictionStateMatches =
    Boolean(
      workflowId &&
      effectiveModelId &&
      predictionState
        ?.workflowId ===
        workflowId &&
      predictionState
        ?.modelId ===
        effectiveModelId
    );


  const selectedPredictionResult =
    predictionStateMatches &&
    predictionState
      ?.status ===
      "ready"
      ? predictionState
          .result
      : null;


  const predictionError =
    predictionStateMatches &&
    predictionState
      ?.status ===
      "error"
      ? predictionState
          .error
      : null;


  function togglePredictionPanel() {
    if (
      !effectiveModelId
    ) {
      return;
    }


    if (
      predictionPanelModelId ===
        effectiveModelId
    ) {
      setPredictionPanelModelId(
        null
      );

      return;
    }


    setPredictionPanelModelId(
      effectiveModelId
    );


    setPredictionFormState({
      modelId:
        effectiveModelId,

      values:
        {},
    });


    setPredictionState(
      null
    );
  }


  function changePredictionValue(
    featureName:
      string,

    value:
      string
  ) {
    if (
      !effectiveModelId
    ) {
      return;
    }


    setPredictionFormState(
      (
        current
      ) => {
        const currentValues =
          current
            ?.modelId ===
            effectiveModelId
            ? current.values
            : {};


        return {
          modelId:
            effectiveModelId,

          values:
            {
              ...currentValues,

              [
                featureName
              ]:
                value,
            },
        };
      }
    );


    setPredictionState(
      null
    );
  }


  async function submitPrediction() {
    if (
      !workflowId ||
      !effectiveModelId ||
      !selectedDetail ||
      predictionSubmitting
    ) {
      return;
    }


    const categoricalFeatures =
      new Set(
        selectedDetail
          .categorical_feature_columns
      );


    const row:
      ModelPredictionRow = {};


    try {
      for (
        const featureName
        of selectedDetail
          .feature_columns
      ) {
        const rawValue =
          predictionFormValues[
            featureName
          ] ??
          "";


        const empty =
          rawValue.trim() ===
          "";


        let value:
          ModelPredictionScalar;


        if (
          categoricalFeatures.has(
            featureName
          )
        ) {
          if (
            empty
          ) {
            if (
              selectedDetail
                .preprocessing
                .categorical_imputation ===
                "most_frequent"
            ) {
              value =
                null;
            } else {
              throw new Error(
                (
                  "La variable " +
                  featureName +
                  " est requise."
                )
              );
            }
          } else {
            value =
              rawValue;
          }
        } else {
          if (
            empty
          ) {
            if (
              selectedDetail
                .preprocessing
                .numeric_imputation ===
                "median"
            ) {
              value =
                null;
            } else {
              throw new Error(
                (
                  "La variable " +
                  featureName +
                  " est requise."
                )
              );
            }
          } else {
            const numericValue =
              Number(
                rawValue
              );


            if (
              !Number.isFinite(
                numericValue
              )
            ) {
              throw new Error(
                (
                  "La variable " +
                  featureName +
                  " doit contenir un nombre fini."
                )
              );
            }


            value =
              numericValue;
          }
        }


        row[
          featureName
        ] = value;
      }
    } catch (
      error
    ) {
      setPredictionState({
        workflowId,

        modelId:
          effectiveModelId,

        status:
          "error",

        result:
          null,

        error:
          errorMessage(
            error
          ),
      });

      return;
    }


    setPredictionSubmitting(
      true
    );


    setPredictionState(
      null
    );


    try {
      const result =
        await predictModelLabRows({
          workflow_id:
            workflowId,

          model_id:
            effectiveModelId,

          rows:
            [
              row,
            ],
        });


      setPredictionState({
        workflowId,

        modelId:
          effectiveModelId,

        status:
          "ready",

        result,

        error:
          null,
      });
    } catch (
      error
    ) {
      setPredictionState({
        workflowId,

        modelId:
          effectiveModelId,

        status:
          "error",

        result:
          null,

        error:
          errorMessage(
            error
          ),
      });
    } finally {
      setPredictionSubmitting(
        false
      );
    }
  }


  /* ========================================================
     MODEL EVALUATION
  ======================================================== */


  const evaluationStateMatches =
    Boolean(
      workflowId &&
      effectiveModelId &&
      evaluationState
        ?.workflowId ===
        workflowId &&
      evaluationState
        ?.modelId ===
        effectiveModelId
    );


  const selectedEvaluation =
    evaluationStateMatches &&
    evaluationState
      ?.status ===
      "ready"
      ? evaluationState
          .evaluation
      : null;


  const evaluationError =
    evaluationStateMatches &&
    evaluationState
      ?.status ===
      "error"
      ? evaluationState
          .error
      : null;


  async function evaluateSelectedModel() {
    if (
      !workflowId ||
      !effectiveModelId ||
      evaluationSubmitting
    ) {
      return;
    }


    setEvaluationSubmitting(
      true
    );


    try {
      const result =
        await evaluateModelLabModel(
          workflowId,
          effectiveModelId
        );


      setEvaluationState({
        workflowId,

        modelId:
          effectiveModelId,

        status:
          "ready",

        evaluation:
          result,

        error:
          null,
      });
    } catch (
      error
    ) {
      setEvaluationState({
        workflowId,

        modelId:
          effectiveModelId,

        status:
          "error",

        evaluation:
          null,

        error:
          errorMessage(
            error
          ),
      });
    } finally {
      setEvaluationSubmitting(
        false
      );
    }
  }


  /* ========================================================
     INVENTORY COUNTERS
  ======================================================== */


  const modelCount =
    inventory
      ?.model_count ??
    0;


  const datasetCount =
    useMemo(
      () => {
        if (
          !inventory
        ) {
          return 0;
        }


        return (
          new Set(
            inventory.models.map(
              (
                model
              ) =>
                model.dataset_id
            )
          ).size
        );
      },
      [
        inventory,
      ]
    );


  const experimentCount =
    useMemo(
      () => {
        if (
          !inventory
        ) {
          return 0;
        }


        return (
          new Set(
            inventory.models
              .map(
                (
                  model
                ) =>
                  model.experiment_id
              )
              .filter(
                (
                  experimentId
                ): experimentId is string =>
                  Boolean(
                    experimentId
                  )
              )
          ).size
        );
      },
      [
        inventory,
      ]
    );


  function handleWorkspaceStepChange(
    step:
      WorkspaceStep
  ): void {
    if (
      workflowId
    ) {
      persistActiveWorkspaceStep(
        workflowId,
        step
      );
    }


    router.push(
      "/"
    );
  }


  /* ========================================================
     RENDER
  ======================================================== */


  return (
    <main
      className={
        workspaceStyles.page
      }
    >
      <header
        className={
          workspaceStyles.header
        }
      >
        <Link
          href="/"
          aria-label="DataLens - workspace"
          className={
            workspaceStyles.brand
          }
        >
          <span
            className={
              workspaceStyles.brandMark
            }
            aria-hidden="true"
          >
            <svg
              className={
                workspaceStyles.brandMarkSvg
              }
              viewBox="0 0 28 28"
              focusable="false"
              aria-hidden="true"
            >
              <path
                className={
                  workspaceStyles.brandMarkOutline
                }
                d="M7 5 H13 C18.2 5 21.5 8.2 21.5 11.1"
              />

              <path
                className={
                  workspaceStyles.brandMarkOutline
                }
                d="M21.5 16.9 C21.5 19.8 18.2 23 13 23 H7 V5"
              />

              <circle
                className={
                  workspaceStyles.brandMarkSignal
                }
                cx="21.5"
                cy="14"
                r="1.75"
              />
            </svg>
          </span>

          <strong>
            DataLens
          </strong>
        </Link>


        <div
          className={
            workspaceStyles.privacyStatus
          }
        >
          <span
            aria-hidden="true"
            className={
              workspaceStyles.statusDot
            }
          />

          <span>
            Traitement local ? donn?es priv?es
          </span>
        </div>
      </header>


      <WorkspaceNavigation
        activeStep={
          null
        }
        onStepChange={
          handleWorkspaceStepChange
        }
        dataReady={
          Boolean(
            workflowId
          )
        }
        reportReady={
          Boolean(
            workflowId
          )
        }
        interventionCount={
          0
        }
        activeAiTool="model-lab"
      />


      <div
        className={
          `${workspaceStyles.shell} ${styles.unifiedShell}`
        }
      >
        <header
          className={
            styles.header
          }
        >
          <div
            className={
              styles.hero
            }
          >
            <div
              className={
                styles.heroCopy
              }
            >
              <p
                className={
                  styles.eyebrow
                }
              >
                Machine Learning · Laboratoire
              </p>


              <h1
                className={
                  styles.title
                }
              >
                Model Lab
              </h1>


              <p
                className={
                  styles.subtitle
                }
              >
                Entraîner, évaluer et surveiller
                des modèles à partir des données
                validées par DataLens.
              </p>
            </div>


            <div
              className={
                styles.authorityCard
              }
            >
              <span
                className={
                  styles.authorityLabel
                }
              >
                Autorité
              </span>

              <strong>
                Contrôle serveur
              </strong>

              <p>
                Les entraînements utilisent uniquement
                les artefacts validés par DataLens.
                Aucun modèle sérialisé n’est chargé
                dans le navigateur.
              </p>
            </div>
          </div>
        </header>


        <section
          aria-label="Contexte Model Lab"
          className={
            styles.contextGrid
          }
        >
          <article
            className={
              styles.contextCard
            }
          >
            <span
              className={
                styles.contextLabel
              }
            >
              Workflow
            </span>

            <strong
              className={
                styles.contextValue
              }
              title={
                workflowId ??
                "Aucun workflow"
              }
            >
              {
                workflowId
                  ? shortIdentifier(
                      workflowId
                    )
                  : "Non sélectionné"
              }
            </strong>

            <span
              className={
                styles.contextMeta
              }
            >
              {
                workflowSourceLabel(
                  workflowSource
                )
              }
            </span>
          </article>


          <article
            className={
              styles.contextCard
            }
          >
            <span
              className={
                styles.contextLabel
              }
            >
              Backend
            </span>

            <strong
              className={
                styles.contextValue
              }
            >
              {
                connectionLabel(
                  connectionState
                )
              }
            </strong>

            <span
              className={
                styles.contextMeta
              }
            >
              Model Lab API v0.1
            </span>
          </article>


          <article
            className={
              styles.contextCard
            }
          >
            <span
              className={
                styles.contextLabel
              }
            >
              Modèles
            </span>

            <strong
              className={
                styles.contextNumber
              }
            >
              {
                connectionState ===
                  "ready"
                  ? modelCount
                  : "?"
              }
            </strong>

            <span
              className={
                styles.contextMeta
              }
            >
              Modèles disponibles
            </span>
          </article>


          <article
            className={
              styles.contextCard
            }
          >
            <span
              className={
                styles.contextLabel
              }
            >
              Sources utilisées
            </span>

            <strong
              className={
                styles.contextNumber
              }
            >
              {
                connectionState ===
                  "ready"
                  ? datasetCount
                  : "?"
              }
            </strong>

            <span
              className={
                styles.contextMeta
              }
            >
              {
                connectionState ===
                  "ready"
                  ? `${experimentCount} ${experimentCount === 1 ? "expérience" : "expériences"}`
                  : "En attente"
              }
            </span>
          </article>
        </section>


        <section
          className={
            styles.workspace
          }
        >
          {
            !workflowId
              ? (
                  <div
                    className={
                      styles.emptyState
                    }
                  >
                    <div
                      className={
                        styles.emptyIcon
                      }
                      aria-hidden="true"
                    >
                      ML
                    </div>


                    <div>
                      <h2>
                        Aucun workflow actif
                      </h2>

                      <p>
                        Le Model Lab travaille
                        dans le contexte d’un
                        workflow Preparation.
                      </p>
                    </div>


                    <Link
                      href="/"
                      className={
                        styles.primaryAction
                      }
                    >
                      Ouvrir le workspace
                    </Link>
                  </div>
                )
              : null
          }


          {
            workflowId &&
            connectionState ===
              "loading"
              ? (
                  <div
                    className={
                      styles.statePanel
                    }
                  >
                    <span
                      className={
                        styles.loadingDot
                      }
                      aria-hidden="true"
                    />

                    <div>
                      <h2>
                        Chargement des modèles
                      </h2>

                      <p>
                        Lecture de l’inventaire
                        serveur.
                      </p>
                    </div>
                  </div>
                )
              : null
          }


          {
            workflowId &&
            connectionState ===
              "error"
              ? (
                  <div
                    className={
                      `${styles.statePanel} ${styles.errorPanel}`
                    }
                  >
                    <div
                      className={
                        styles.errorMark
                      }
                      aria-hidden="true"
                    >
                      !
                    </div>

                    <div>
                      <h2>
                        Model Lab indisponible
                      </h2>

                      <p>
                        {
                          loadError ??
                          "Une erreur inconnue est survenue."
                        }
                      </p>
                    </div>
                  </div>
                )
              : null
          }


          {
            workflowId &&
            connectionState ===
              "ready" &&
            inventory
              ? (
                  <div
                    className={
                      styles.trainingToolbar
                    }
                  >
                    <div>
                      <p
                        className={
                          styles.sectionEyebrow
                        }
                      >
                        Entraînement
                      </p>

                      <strong>
                        Créer un modèle
                      </strong>

                      <span>
                        Source validée par la préparation DataLens.
                      </span>
                    </div>


                    <button
                      type="button"
                      className={
                        styles.trainingOpenButton
                      }
                      onClick={
                        openTrainingPanel
                      }
                    >
                      Entraîner un modèle
                    </button>
                  </div>
                )
              : null
          }


          {
            trainingOpen &&
            workflowId
              ? (
                  <section
                    className={
                      styles.trainingPanel
                    }
                    aria-label="Configuration de l'entraînement"
                  >
                    <div
                      className={
                        styles.trainingPanelHeader
                      }
                    >
                      <div>
                        <p
                          className={
                            styles.sectionEyebrow
                          }
                        >
                          Nouveau modèle
                        </p>

                        <h2>
                          Configurer l’entraînement
                        </h2>

                        <p>
                          Définissez les données, les variables,
                          le modèle et les paramètres de validation.
                          La source reste contrôlée par le serveur.
                        </p>
                      </div>


                      <button
                        type="button"
                        className={
                          styles.trainingCloseButton
                        }
                        onClick={
                          closeTrainingPanel
                        }
                        disabled={
                          trainingSubmitting
                        }
                      >
                        Fermer
                      </button>
                    </div>


                    {
                      trainingContextLoading
                        ? (
                            <div
                              className={
                                styles.trainingState
                              }
                            >
                              <span
                                className={
                                  styles.loadingDot
                                }
                                aria-hidden="true"
                              />

                              <span>
                                Chargement du contexte d’entraînement…
                              </span>
                            </div>
                          )
                        : null
                    }


                    {
                      trainingContextError
                        ? (
                            <div
                              className={
                                `${styles.trainingState} ${styles.errorPanel}`
                              }
                            >
                              {
                                trainingContextError
                              }
                            </div>
                          )
                        : null
                    }


                    {
                      trainingContext &&
                      selectedTrainingDataset
                        ? (
                            <form
                              className={
                                styles.trainingForm
                              }
                              onSubmit={
                                (
                                  event
                                ) => {
                                  event.preventDefault();

                                  void submitTraining();
                                }
                              }
                            >
                              <div
                                className={
                                  styles.trainingWorkflowSteps
                                }
                                aria-label="Étapes de configuration du modèle"
                              >
                                <div
                                  className={
                                    styles.trainingWorkflowStep
                                  }
                                >
                                  <span>
                                    1
                                  </span>

                                  <div>
                                    <strong>
                                      Données
                                    </strong>

                                    <small>
                                      Dataset · cible
                                    </small>
                                  </div>
                                </div>


                                <div
                                  className={
                                    styles.trainingWorkflowStep
                                  }
                                >
                                  <span>
                                    2
                                  </span>

                                  <div>
                                    <strong>
                                      Variables
                                    </strong>

                                    <small>
                                      Signaux explicatifs
                                    </small>
                                  </div>
                                </div>


                                <div
                                  className={
                                    styles.trainingWorkflowStep
                                  }
                                >
                                  <span>
                                    3
                                  </span>

                                  <div>
                                    <strong>
                                      Modèle
                                    </strong>

                                    <small>
                                      Problème · estimateur
                                    </small>
                                  </div>
                                </div>


                                <div
                                  className={
                                    styles.trainingWorkflowStep
                                  }
                                >
                                  <span>
                                    4
                                  </span>

                                  <div>
                                    <strong>
                                      Validation
                                    </strong>

                                    <small>
                                      Split · reproductibilité
                                    </small>
                                  </div>
                                </div>
                              </div>


                              <div
                                className={
                                  styles.trainingFormGrid
                                }
                              >
                                <label
                                  className={
                                    styles.trainingField
                                  }
                                >
                                  <span>
                                    Dataset
                                  </span>

                                  <select
                                    value={
                                      selectedTrainingDataset
                                        .dataset_id
                                    }
                                    onChange={
                                      (
                                        event
                                      ) => {
                                        changeTrainingDataset(
                                          event.target.value
                                        );
                                      }
                                    }
                                    disabled={
                                      trainingSubmitting
                                    }
                                  >
                                    {
                                      trainingContext
                                        .datasets
                                        .map(
                                          (
                                            dataset
                                          ) => (
                                            <option
                                              key={
                                                dataset.dataset_id
                                              }
                                              value={
                                                dataset.dataset_id
                                              }
                                            >
                                              {
                                                dataset.filename
                                              }
                                            </option>
                                          )
                                        )
                                    }
                                  </select>

                                  <small>
                                    {
                                      selectedTrainingDataset.row_count
                                    } lignes · {
                                      selectedTrainingDataset.column_count
                                    } colonnes
                                  </small>
                                </label>


                                <label
                                  className={
                                    styles.trainingField
                                  }
                                >
                                  <span>
                                    Type de problème
                                  </span>

                                  <select
                                    value={
                                      trainingProblemType
                                    }
                                    onChange={
                                      (
                                        event
                                      ) => {
                                        changeTrainingProblemType(
                                          event.target.value as
                                            ModelLabProblemType
                                        );
                                      }
                                    }
                                    disabled={
                                      trainingSubmitting
                                    }
                                  >
                                    <option
                                      value="classification"
                                    >
                                      Classification
                                    </option>

                                    <option
                                      value="regression"
                                    >
                                      Régression
                                    </option>
                                  </select>

                                  <small>
                                    Détermine les estimateurs et les cibles compatibles.
                                  </small>
                                </label>


                                <label
                                  className={
                                    styles.trainingField
                                  }
                                >
                                  <span>
                                    Variable cible
                                  </span>

                                  <select
                                    value={
                                      trainingTargetColumn
                                    }
                                    onChange={
                                      (
                                        event
                                      ) => {
                                        changeTrainingTarget(
                                          event.target.value
                                        );
                                      }
                                    }
                                    disabled={
                                      trainingSubmitting
                                    }
                                  >
                                    <option
                                      value=""
                                    >
                                      Choisir une cible
                                    </option>

                                    {
                                      eligibleTargetColumns.map(
                                        (
                                          column
                                        ) => (
                                          <option
                                            key={
                                              column.name
                                            }
                                            value={
                                              column.name
                                            }
                                          >
                                            {
                                              column.name
                                            } · {
                                              trainingColumnKindLabel(
                                                column.kind
                                              )
                                            }
                                          </option>
                                        )
                                      )
                                    }
                                  </select>

                                  <small>
                                    La cible est automatiquement exclue des variables explicatives.
                                  </small>
                                </label>


                                <label
                                  className={
                                    styles.trainingField
                                  }
                                >
                                  <span>
                                    Estimateur
                                  </span>

                                  <select
                                    value={
                                      trainingEstimatorKey
                                    }
                                    onChange={
                                      (
                                        event
                                      ) => {
                                        setTrainingEstimatorKey(
                                          event.target.value
                                        );

                                        setTrainingSubmitError(
                                          null
                                        );
                                      }
                                    }
                                    disabled={
                                      trainingSubmitting
                                    }
                                  >
                                    {
                                      estimatorOptions(
                                        trainingProblemType
                                      ).map(
                                        (
                                          estimator
                                        ) => (
                                          <option
                                            key={
                                              estimator.value
                                            }
                                            value={
                                              estimator.value
                                            }
                                          >
                                            {
                                              estimator.label
                                            }
                                          </option>
                                        )
                                      )
                                    }
                                  </select>

                                  <small>
                                    Hyperparamètres déterministes DataLens v0.1.
                                  </small>
                                </label>
                              </div>


                              <div
                                className={
                                  styles.trainingFeatures
                                }
                              >
                                <div
                                  className={
                                    styles.trainingFeaturesHeader
                                  }
                                >
                                  <div>
                                    <span>
                                      Variables explicatives
                                    </span>

                                    <small>
                                      Sélectionnez au moins une variable.
                                    </small>
                                  </div>

                                  <strong>
                                    {
                                      trainingFeatureColumns.length
                                    } {" "}
                                    {
                                      trainingFeatureColumns.length ===
                                        1
                                        ? "sélectionnée"
                                        : "sélectionnées"
                                    }
                                  </strong>
                                </div>


                                <div
                                  className={
                                    styles.trainingFeatureGrid
                                  }
                                >
                                  {
                                    eligibleFeatureColumns.map(
                                      (
                                        column
                                      ) => {
                                        const checked =
                                          selectedTrainingFeatureSet.has(
                                            column.name
                                          );


                                        return (
                                          <label
                                            key={
                                              column.name
                                            }
                                            className={
                                              `${styles.trainingFeatureOption} ${
                                                checked
                                                  ? styles.trainingFeatureOptionSelected
                                                  : ""
                                              }`
                                            }
                                          >
                                            <input
                                              type="checkbox"
                                              checked={
                                                checked
                                              }
                                              onChange={
                                                () => {
                                                  toggleTrainingFeature(
                                                    column.name
                                                  );
                                                }
                                              }
                                              disabled={
                                                trainingSubmitting
                                              }
                                            />

                                            <span>
                                              <strong>
                                                {
                                                  column.name
                                                }
                                              </strong>

                                              <small>
                                                {
                                                  trainingColumnKindLabel(
                                                    column.kind
                                                  )
                                                }
                                                {
                                                  column.nullable
                                                    ? " · valeurs manquantes"
                                                    : ""
                                                }
                                              </small>
                                            </span>
                                          </label>
                                        );
                                      }
                                    )
                                  }
                                </div>


                                {
                                  identifierTrainingColumns.length >
                                    0
                                    ? (
                                        <p
                                          className={
                                            styles.trainingFeatureNote
                                          }
                                        >
                                          Identifiants détectés par DataLens et
                                          exclus automatiquement des cibles et
                                          variables explicatives :{" "}
                                          {
                                            identifierTrainingColumns
                                              .map(
                                                (
                                                  column
                                                ) =>
                                                  column.name
                                              )
                                              .join(
                                                ", "
                                              )
                                          }.
                                        </p>
                                      )
                                    : null
                                }


                                {
                                  selectedTrainingDataset
                                    .columns
                                    .some(
                                      (
                                        column
                                      ) =>
                                        !isDirectMLFeatureKind(
                                          column.kind
                                        )
                                    )
                                    ? (
                                        <p
                                          className={
                                            styles.trainingFeatureNote
                                          }
                                        >
                                          Les colonnes date/heure ou non supportées
                                          doivent d’abord être transformées pendant
                                          la préparation.
                                        </p>
                                      )
                                    : null
                                }
                              </div>


                              <div
                                className={
                                  styles.trainingValidationHeader
                                }
                              >
                                <div>
                                  <span>
                                    Validation
                                  </span>

                                  <strong>
                                    Paramètres reproductibles
                                  </strong>
                                </div>

                                <small>
                                  Configuration déterministe appliquée à cet entraînement.
                                </small>
                              </div>


                              <div
                                className={
                                  styles.trainingFormGrid
                                }
                              >
                                <label
                                  className={
                                    styles.trainingField
                                  }
                                >
                                  <span>
                                    Séparation train / test
                                  </span>

                                  <select
                                    value={
                                      trainingSplitStrategy
                                    }
                                    onChange={
                                      (
                                        event
                                      ) => {
                                        changeTrainingSplitStrategy(
                                          event.target.value as
                                            ModelTrainingSplitStrategy
                                        );
                                      }
                                    }
                                    disabled={
                                      trainingSubmitting
                                    }
                                  >
                                    <option
                                      value=""
                                    >
                                      Choisir la méthode
                                    </option>

                                    <option
                                      value="holdout"
                                    >
                                      Par lignes
                                    </option>

                                    <option
                                      value="time_holdout"
                                      disabled={
                                        eligibleTrainingTimeColumns.length ===
                                          0
                                      }
                                    >
                                      Chronologique
                                    </option>

                                    <option
                                      value="group_holdout"
                                      disabled={
                                        eligibleTrainingGroupColumns.length ===
                                          0
                                      }
                                    >
                                      Par entité
                                    </option>

                                    <option
                                      value="purged_group_time_holdout"
                                      disabled={
                                        eligibleTrainingGroupColumns.length ===
                                          0
                                        ||
                                        eligibleTrainingTimeColumns.length ===
                                          0
                                      }
                                    >
                                      Chronologique + purge entités
                                    </option>
                                  </select>

                                  <small>
                                    {
                                      trainingSplitIsPurgedGroupTime
                                        ? (
                                            "Le test conserve les observations futures. " +
                                            "Toute entité présente dans ce futur est retirée " +
                                            "de l’historique d’entraînement pour éviter une " +
                                            "fuite d’entité entre train et test."
                                          )
                                        : (
                                            trainingSplitStrategy ===
                                              "group_holdout"
                                              ? (
                                                  "Le mode par entité empêche une même entité " +
                                                  "d’apparaître dans le train et le test."
                                                )
                                              : (
                                                  trainingSplitStrategy ===
                                                    "time_holdout"
                                                    ? (
                                                        "Le mode chronologique entraîne sur le passé " +
                                                        "et conserve les observations futures pour le test."
                                                      )
                                                    : (
                                                        "Le mode par lignes applique un holdout " +
                                                        "déterministe 80 / 20."
                                                      )
                                                )
                                          )
                                    }
                                  </small>
                                </label>


                                {
                                  trainingSplitUsesGroup
                                    ? (
                                        <label
                                          className={
                                            styles.trainingField
                                          }
                                        >
                                          <span>
                                            Colonne d’entité
                                          </span>

                                          <select
                                            value={
                                              trainingGroupColumn
                                            }
                                            onChange={
                                              (
                                                event
                                              ) => {
                                                setTrainingGroupColumn(
                                                  event.target.value
                                                );

                                                setTrainingSubmitError(
                                                  null
                                                );
                                              }
                                            }
                                            disabled={
                                              trainingSubmitting
                                            }
                                          >
                                            <option
                                              value=""
                                            >
                                              Choisir une entité
                                            </option>

                                            {
                                              eligibleTrainingGroupColumns.map(
                                                (
                                                  column
                                                ) => (
                                                  <option
                                                    key={
                                                      column.name
                                                    }
                                                    value={
                                                      column.name
                                                    }
                                                  >
                                                    {
                                                      column.name
                                                    }
                                                  </option>
                                                )
                                              )
                                            }
                                          </select>

                                          <small>
                                            Seuls les identifiants de
                                            référence répétés,
                                            sans valeur manquante et validés
                                            par le serveur sont proposés.
                                          </small>
                                        </label>
                                      )
                                    : null
                                }
                                {
                                  trainingSplitUsesTime
                                    ? (
                                        <label
                                          className={
                                            styles.trainingField
                                          }
                                        >
                                          <span>
                                            Colonne temporelle
                                          </span>

                                          <select
                                            value={
                                              trainingTimeColumn
                                            }
                                            onChange={
                                              (
                                                event
                                              ) => {
                                                setTrainingTimeColumn(
                                                  event.target.value
                                                );

                                                setTrainingSubmitError(
                                                  null
                                                );
                                              }
                                            }
                                            disabled={
                                              trainingSubmitting
                                            }
                                          >
                                            <option
                                              value=""
                                            >
                                              Choisir une date d’observation
                                            </option>

                                            {
                                              eligibleTrainingTimeColumns.map(
                                                (
                                                  column
                                                ) => (
                                                  <option
                                                    key={
                                                      column.name
                                                    }
                                                    value={
                                                      column.name
                                                    }
                                                  >
                                                    {
                                                      column.name
                                                    }
                                                  </option>
                                                )
                                              )
                                            }
                                          </select>

                                          <small>
                                            Seules les colonnes datetime
                                            non nulles et validées par le
                                            serveur sont proposées.
                                          </small>
                                        </label>
                                      )
                                    : null
                                }


                              </div>


                              <div
                                className={
                                  styles.trainingContractSummary
                                }
                              >
                                <div>
                                  <span>
                                    Révision préparation
                                  </span>

                                  <strong>
                                    {
                                      trainingContext
                                        .preparation_session_revision
                                    }
                                  </strong>
                                </div>

                                <div>
                                  <span>
                                    Split
                                  </span>

                                  <strong>
                                    {
                                      trainingSplitIsPurgedGroupTime
                                        ? "Passé / futur + purge entités"
                                        : (
                                            trainingSplitStrategy ===
                                              "group_holdout"
                                              ? "80 / 20 des entités"
                                              : (
                                                  trainingSplitStrategy ===
                                                    "time_holdout"
                                                    ? "Passé / futur"
                                                    : (
                                                        trainingSplitStrategy ===
                                                          "holdout"
                                                          ? "80 / 20 des lignes"
                                                          : "À choisir"
                                                      )
                                                )
                                          )
                                    }
                                  </strong>
                                </div>

                                <div>
                                  <span>
                                    Entité
                                  </span>

                                  <strong>
                                    {
                                      trainingSplitUsesGroup
                                        ? (
                                            trainingGroupColumn ||
                                            "À choisir"
                                          )
                                        : "Aucune"
                                    }
                                  </strong>
                                </div>

                                <div>
                                  <span>
                                    Temps
                                  </span>

                                  <strong>
                                    {
                                      trainingSplitUsesTime
                                        ? (
                                            trainingTimeColumn ||
                                            "À choisir"
                                          )
                                        : "Aucun"
                                    }
                                  </strong>
                                </div>

                                <div>
                                  <span>
                                    Seed
                                  </span>

                                  <strong>
                                    42
                                  </strong>
                                </div>

                                <div>
                                  <span>
                                    Stratification
                                  </span>

                                  <strong>
                                    {
                                      trainingSplitStrategy ===
                                        "holdout"
                                      &&
                                      trainingProblemType ===
                                        "classification"
                                        ? "Oui"
                                        : "Non"
                                    }
                                  </strong>
                                </div>
                              </div>


                              {
                                trainingSubmitError
                                  ? (
                                      <div
                                        className={
                                          styles.trainingSubmitError
                                        }
                                      >
                                        {
                                          trainingSubmitError
                                        }
                                      </div>
                                    )
                                  : null
                              }


                              <div
                                className={
                                  styles.trainingActions
                                }
                              >
                                <button
                                  type="button"
                                  className={
                                    styles.trainingSecondaryButton
                                  }
                                  onClick={
                                    closeTrainingPanel
                                  }
                                  disabled={
                                    trainingSubmitting
                                  }
                                >
                                  Annuler
                                </button>

                                <button
                                  type="submit"
                                  className={
                                    styles.trainingPrimaryButton
                                  }
                                  disabled={
                                    !trainingCanSubmit
                                  }
                                >
                                  {
                                    trainingSubmitting
                                      ? "Entraînement en cours…"
                                      : "Entraîner le modèle"
                                  }
                                </button>
                              </div>
                            </form>
                          )
                        : null
                    }
                  </section>
                )
              : null
          }


          {
            workflowId &&
            connectionState ===
              "ready" &&
            inventory &&
            modelCount ===
              0
              ? (
                  <div
                    className={
                      styles.emptyInventory
                    }
                  >
                    <p
                      className={
                        styles.sectionEyebrow
                      }
                    >
                      Inventaire
                    </p>

                    <h2>
                      Aucun modèle entraîné
                    </h2>

                    <p>
                      Ce workflow ne possède
                      encore aucun modèle entraîné
                      exploitable.
                    </p>
                  </div>
                )
              : null
          }


          {
            workflowId &&
            connectionState ===
              "ready" &&
            inventory &&
            modelCount >
              0
              ? (
                  <div
                    className={
                      styles.modelWorkspace
                    }
                  >
                    <aside
                      className={
                        styles.modelSidebar
                      }
                    >
                      <div
                        className={
                          styles.sidebarHeader
                        }
                      >
                        <div>
                          <p
                            className={
                              styles.sectionEyebrow
                            }
                          >
                            Inventaire
                          </p>

                          <h2>
                            {
                              modelCount ===
                                1
                                ? "1 modèle"
                                : `${modelCount} modèles`
                            }
                          </h2>
                        </div>


                        <span
                          className={
                            styles.readyBadge
                          }
                        >
                          API connectée
                        </span>
                      </div>


                      <div
                        className={
                          styles.modelList
                        }
                      >
                        {
                          inventory.models.map(
                            (
                              model
                            ) => {
                              const active =
                                model.model_id ===
                                selectedModel
                                  ?.model_id;


                              const metric =
                                primaryMetricEntry(
                                  model
                                );


                              return (
                                <button
                                  key={
                                    model.model_id
                                  }
                                  type="button"
                                  className={
                                    `${styles.modelCard} ${
                                      active
                                        ? styles.modelCardActive
                                        : ""
                                    }`
                                  }
                                  aria-pressed={
                                    active
                                  }
                                  onClick={
                                    () => {
                                      setSelectedModelId(
                                        model.model_id
                                      );
                                    }
                                  }
                                >
                                  <div
                                    className={
                                      styles.modelCardTop
                                    }
                                  >
                                    <span
                                      className={
                                        styles.problemBadge
                                      }
                                    >
                                      {
                                        problemTypeLabel(
                                          model.problem_type
                                        )
                                      }
                                    </span>

                                    {
                                      model.has_experiment_provenance
                                        ? (
                                            <span
                                              className={
                                                styles.provenanceDot
                                              }
                                              title="Provenance expérimentale disponible"
                                            />
                                          )
                                        : null
                                    }
                                  </div>


                                  <strong
                                    className={
                                      styles.modelName
                                    }
                                  >
                                    {
                                      estimatorLabel(
                                        model.estimator_key
                                      )
                                    }
                                  </strong>


                                  <span
                                    className={
                                      styles.modelTarget
                                    }
                                  >
                                    Cible · {
                                      model.target_column
                                    }
                                  </span>


                                  {
                                    metric
                                      ? (
                                          <div
                                            className={
                                              styles.cardMetric
                                            }
                                          >
                                            <span>
                                              {
                                                metricLabel(
                                                  metric[
                                                    0
                                                  ]
                                                )
                                              }
                                            </span>

                                            <strong>
                                              {
                                                formatMetric(
                                                  metric[
                                                    1
                                                  ]
                                                )
                                              }
                                            </strong>
                                          </div>
                                        )
                                      : null
                                  }


                                  <span
                                    className={
                                      styles.modelDate
                                    }
                                  >
                                    {
                                      formatDate(
                                        model.created_at_utc
                                      )
                                    }
                                  </span>
                                </button>
                              );
                            }
                          )
                        }
                      </div>
                    </aside>


                    <section
                      className={
                        styles.modelDetail
                      }
                    >
                      {
                        selectedModel
                          ? (
                              <>
                                <div
                                  className={
                                    styles.detailHeader
                                  }
                                >
                                  <div>
                                    <p
                                      className={
                                        styles.sectionEyebrow
                                      }
                                    >
                                      Modèle sélectionné
                                    </p>

                                    <h2
                                      className={
                                        styles.detailTitle
                                      }
                                    >
                                      {
                                        estimatorLabel(
                                          selectedModel.estimator_key
                                        )
                                      }
                                    </h2>

                                    <p
                                      className={
                                        styles.detailSubtitle
                                      }
                                    >
                                      {
                                        problemTypeLabel(
                                          selectedModel.problem_type
                                        )
                                      }
                                      {" · Cible "}
                                      <strong>
                                        {
                                          selectedModel.target_column
                                        }
                                      </strong>
                                    </p>
                                  </div>


                                  <div
                                    className={
                                      styles.detailHeaderActions
                                    }
                                  >
                                    <button
                                      type="button"
                                      className={
                                        styles.evaluateButton
                                      }
                                      onClick={
                                        () => {
                                          void evaluateSelectedModel();
                                        }
                                      }
                                      disabled={
                                        evaluationSubmitting
                                      }
                                    >
                                      {
                                        evaluationSubmitting
                                          ? "Évaluation en cours…"
                                          : (
                                              selectedEvaluation
                                                ? "Réévaluer"
                                                : "Évaluer le modèle"
                                            )
                                      }
                                    </button>


                                    <button
                                      type="button"
                                      className={
                                        predictionPanelOpen
                                          ? `${styles.predictButton} ${styles.predictButtonActive}`
                                          : styles.predictButton
                                      }
                                      onClick={
                                        togglePredictionPanel
                                      }
                                      disabled={
                                        !selectedDetail ||
                                        predictionSubmitting
                                      }
                                    >
                                      {
                                        predictionPanelOpen
                                          ? "Fermer la prédiction"
                                          : "Tester une prédiction"
                                      }
                                    </button>


                                    <div
                                      className={
                                        styles.modelIdentity
                                      }
                                      title={
                                        selectedModel.model_id
                                      }
                                    >
                                      {
                                        shortIdentifier(
                                          selectedModel.model_id
                                        )
                                      }
                                    </div>
                                  </div>
                                </div>


                                <div
                                  className={
                                    styles.metricsGrid
                                  }
                                >
                                  {
                                    Object.entries(
                                      selectedModel.metrics
                                    ).map(
                                      (
                                        [
                                          name,
                                          value,
                                        ]
                                      ) => (
                                        <article
                                          key={
                                            name
                                          }
                                          className={
                                            styles.metricCard
                                          }
                                        >
                                          <span>
                                            {
                                              metricLabel(
                                                name
                                              )
                                            }
                                          </span>

                                          <strong>
                                            {
                                              formatMetric(
                                                value
                                              )
                                            }
                                          </strong>
                                        </article>
                                      )
                                    )
                                  }
                                </div>


                                {
                                  workflowId
                                    ? (
                                        <ModelObservabilityPanel
                                          workflowId={
                                            workflowId
                                          }
                                          modelId={
                                            selectedModel.model_id
                                          }
                                        />
                                      )
                                    : null
                                }


                                {
                                  predictionPanelOpen &&
                                  selectedDetail
                                    ? (
                                        <section
                                          className={
                                            styles.predictionPanel
                                          }
                                        >
                                          <div
                                            className={
                                              styles.predictionHeader
                                            }
                                          >
                                            <div>
                                              <p
                                                className={
                                                  styles.sectionEyebrow
                                                }
                                              >
                                                Prédiction
                                              </p>

                                              <h3>
                                                Tester une observation
                                              </h3>

                                              <p>
                                                Renseignez exactement les variables
                                                apprises par ce Model Artifact.
                                                La prédiction est calculée par
                                                le modèle restauré côté serveur.
                                              </p>
                                            </div>


                                            <span
                                              className={
                                                styles.predictionMethod
                                              }
                                            >
                                              trusted_native_predict
                                            </span>
                                          </div>


                                          <form
                                            className={
                                              styles.predictionForm
                                            }
                                            onSubmit={
                                              (
                                                event
                                              ) => {
                                                event.preventDefault();

                                                void submitPrediction();
                                              }
                                            }
                                          >
                                            <div
                                              className={
                                                styles.predictionFeatureGrid
                                              }
                                            >
                                              {
                                                selectedDetail
                                                  .feature_columns
                                                  .map(
                                                    (
                                                      featureName
                                                    ) => {
                                                      const categorical =
                                                        selectedDetail
                                                          .categorical_feature_columns
                                                          .includes(
                                                            featureName
                                                          );


                                                      return (
                                                        <label
                                                          key={
                                                            featureName
                                                          }
                                                          className={
                                                            styles.predictionField
                                                          }
                                                        >
                                                          <span>
                                                            {
                                                              featureName
                                                            }
                                                          </span>

                                                          <input
                                                            type={
                                                              categorical
                                                                ? "text"
                                                                : "number"
                                                            }
                                                            step={
                                                              categorical
                                                                ? undefined
                                                                : "any"
                                                            }
                                                            inputMode={
                                                              categorical
                                                                ? "text"
                                                                : "decimal"
                                                            }
                                                            value={
                                                              predictionFormValues[
                                                                featureName
                                                              ] ??
                                                              ""
                                                            }
                                                            placeholder={
                                                              categorical
                                                                ? "Valeur catégorielle"
                                                                : "Valeur numérique"
                                                            }
                                                            onChange={
                                                              (
                                                                event
                                                              ) => {
                                                                changePredictionValue(
                                                                  featureName,
                                                                  event.target.value
                                                                );
                                                              }
                                                            }
                                                            disabled={
                                                              predictionSubmitting
                                                            }
                                                          />

                                                          <small>
                                                            {
                                                              categorical
                                                                ? "Catégorielle"
                                                                : "Numérique"
                                                            }
                                                            {
                                                              categorical
                                                                ? (
                                                                    selectedDetail
                                                                      .preprocessing
                                                                      .categorical_imputation ===
                                                                      "most_frequent"
                                                                      ? " · vide autorisé"
                                                                      : " · requise"
                                                                  )
                                                                : (
                                                                    selectedDetail
                                                                      .preprocessing
                                                                      .numeric_imputation ===
                                                                      "median"
                                                                      ? " · vide autorisé"
                                                                      : " · requise"
                                                                  )
                                                            }
                                                          </small>
                                                        </label>
                                                      );
                                                    }
                                                  )
                                              }
                                            </div>


                                            <p
                                              className={
                                                styles.predictionPrivacyNote
                                              }
                                            >
                                              Les catégories d’entraînement ne
                                              sont pas exposées au navigateur.
                                              Les valeurs saisies sont envoyées
                                              uniquement pour cette requête de prédiction.
                                            </p>


                                            {
                                              predictionError
                                                ? (
                                                    <div
                                                      className={
                                                        styles.predictionError
                                                      }
                                                    >
                                                      {
                                                        predictionError
                                                      }
                                                    </div>
                                                  )
                                                : null
                                            }


                                            {
                                              selectedPredictionResult
                                                ? (
                                                    <div
                                                      className={
                                                        styles.predictionResult
                                                      }
                                                    >
                                                      <div>
                                                        <span>
                                                          Cible prédite
                                                        </span>

                                                        <strong>
                                                          {
                                                            selectedPredictionResult
                                                              .target_column
                                                          }
                                                        </strong>
                                                      </div>


                                                      <div
                                                        className={
                                                          styles.predictionResultValue
                                                        }
                                                      >
                                                        <span>
                                                          Résultat
                                                        </span>

                                                        <strong>
                                                          {
                                                            predictionDisplayValue(
                                                              selectedPredictionResult
                                                                .predictions[
                                                                  0
                                                                ]
                                                            )
                                                          }
                                                        </strong>
                                                      </div>


                                                      <small>
                                                        1 observation · {
                                                          selectedPredictionResult.method
                                                        }
                                                      </small>
                                                    </div>
                                                  )
                                                : null
                                            }


                                            <div
                                              className={
                                                styles.predictionActions
                                              }
                                            >
                                              <button
                                                type="button"
                                                className={
                                                  styles.predictionSecondaryButton
                                                }
                                                onClick={
                                                  () => {
                                                    if (
                                                      effectiveModelId
                                                    ) {
                                                      setPredictionFormState({
                                                        modelId:
                                                          effectiveModelId,

                                                        values:
                                                          {},
                                                      });

                                                      setPredictionState(
                                                        null
                                                      );
                                                    }
                                                  }
                                                }
                                                disabled={
                                                  predictionSubmitting
                                                }
                                              >
                                                Effacer
                                              </button>


                                              <button
                                                type="submit"
                                                className={
                                                  styles.predictionPrimaryButton
                                                }
                                                disabled={
                                                  predictionSubmitting
                                                }
                                              >
                                                {
                                                  predictionSubmitting
                                                    ? "Prédiction en cours…"
                                                    : "Lancer la prédiction"
                                                }
                                              </button>
                                            </div>
                                          </form>
                                        </section>
                                      )
                                    : null
                                }


                                {
                                  evaluationError
                                    ? (
                                        <div
                                          className={
                                            styles.evaluationError
                                          }
                                        >
                                          {
                                            evaluationError
                                          }
                                        </div>
                                      )
                                    : null
                                }


                                {
                                  selectedEvaluation
                                    ? (
                                        <section
                                          className={
                                            styles.evaluationPanel
                                          }
                                        >
                                          <div
                                            className={
                                              styles.evaluationHeader
                                            }
                                          >
                                            <div>
                                              <p
                                                className={
                                                  styles.sectionEyebrow
                                                }
                                              >
                                                évaluation serveur
                                              </p>

                                              <h3>
                                                Performance vérifiée
                                              </h3>
                                            </div>


                                            <span
                                              className={
                                                selectedEvaluation
                                                  .baseline_comparison
                                                  .beats_baseline
                                                  ? styles.baselinePositive
                                                  : styles.baselineNegative
                                              }
                                            >
                                              {
                                                selectedEvaluation
                                                  .baseline_comparison
                                                  .beats_baseline
                                                  ? "Bat la baseline"
                                                  : "Ne bat pas la baseline"
                                              }
                                            </span>
                                          </div>


                                          <div
                                            className={
                                              styles.evaluationSummaryGrid
                                            }
                                          >
                                            <article>
                                              <span>
                                                Métrique principale
                                              </span>

                                              <strong>
                                                {
                                                  metricLabel(
                                                    selectedEvaluation
                                                      .baseline_comparison
                                                      .primary_metric
                                                  )
                                                }
                                              </strong>
                                            </article>


                                            <article>
                                              <span>
                                                Modèle
                                              </span>

                                              <strong>
                                                {
                                                  formatMetric(
                                                    selectedEvaluation
                                                      .baseline_comparison
                                                      .model_primary_metric_value
                                                  )
                                                }
                                              </strong>
                                            </article>


                                            <article>
                                              <span>
                                                Baseline
                                              </span>

                                              <strong>
                                                {
                                                  formatMetric(
                                                    selectedEvaluation
                                                      .baseline_comparison
                                                      .baseline_primary_metric_value
                                                  )
                                                }
                                              </strong>
                                            </article>


                                            <article>
                                              <span>
                                                Gain absolu
                                              </span>

                                              <strong>
                                                {
                                                  formatMetric(
                                                    selectedEvaluation
                                                      .baseline_comparison
                                                      .absolute_improvement
                                                  )
                                                }
                                              </strong>
                                            </article>
                                          </div>


                                          <div
                                            className={
                                              styles.evaluationColumns
                                            }
                                          >
                                            <section
                                              className={
                                                styles.evaluationBlock
                                              }
                                            >
                                              <div
                                                className={
                                                  styles.evaluationBlockHeader
                                                }
                                              >
                                                <span>
                                                  Baseline
                                                </span>

                                                <strong>
                                                  {
                                                    selectedEvaluation
                                                      .baseline
                                                      .strategy ===
                                                      "majority_train_class"
                                                      ? "Classe majoritaire"
                                                      : "Moyenne de la cible"
                                                  }
                                                </strong>
                                              </div>


                                              <div
                                                className={
                                                  styles.evaluationMetricsList
                                                }
                                              >
                                                {
                                                  Object.entries(
                                                    selectedEvaluation
                                                      .baseline
                                                      .metrics
                                                  ).map(
                                                    (
                                                      [
                                                        name,
                                                        value,
                                                      ]
                                                    ) => (
                                                      <div
                                                        key={
                                                          name
                                                        }
                                                      >
                                                        <span>
                                                          {
                                                            metricLabel(
                                                              name
                                                            )
                                                          }
                                                        </span>

                                                        <strong>
                                                          {
                                                            formatMetric(
                                                              value
                                                            )
                                                          }
                                                        </strong>
                                                      </div>
                                                    )
                                                  )
                                                }
                                              </div>
                                            </section>


                                            <section
                                              className={
                                                styles.evaluationBlock
                                              }
                                            >
                                              <div
                                                className={
                                                  styles.evaluationBlockHeader
                                                }
                                              >
                                                <span>
                                                  Sélection
                                                </span>

                                                <strong>
                                                  {
                                                    selectedEvaluation
                                                      .selection_evidence
                                                      .status ===
                                                      "verified_selected"
                                                      ? "Sélection vérifiée"
                                                      : "Modèle autonome"
                                                  }
                                                </strong>
                                              </div>


                                              <dl
                                                className={
                                                  styles.evaluationDefinitionList
                                                }
                                              >
                                                <div>
                                                  <dt>
                                                    Source
                                                  </dt>

                                                  <dd>
                                                    {
                                                      selectedEvaluation
                                                        .selection_evidence
                                                        .source
                                                    }
                                                  </dd>
                                                </div>

                                                <div>
                                                  <dt>
                                                    Portée
                                                  </dt>

                                                  <dd>
                                                    {
                                                      selectedEvaluation
                                                        .selection_evidence
                                                        .metric_scope
                                                    }
                                                  </dd>
                                                </div>
                                              </dl>
                                            </section>
                                          </div>


                                          {
                                            selectedEvaluation
                                              .classification_diagnostics
                                              ? (
                                                  <section
                                                    className={
                                                      styles.evaluationBlock
                                                    }
                                                  >
                                                    <div
                                                      className={
                                                        styles.evaluationBlockHeader
                                                      }
                                                    >
                                                      <span>
                                                        Diagnostics par classe
                                                      </span>

                                                      <strong>
                                                        {
                                                          selectedEvaluation
                                                            .classification_diagnostics
                                                            .evaluation_rows
                                                        } lignes holdout
                                                      </strong>
                                                    </div>


                                                    <div
                                                      className={
                                                        styles.classDiagnosticsGrid
                                                      }
                                                    >
                                                      {
                                                        selectedEvaluation
                                                          .classification_diagnostics
                                                          .per_class
                                                          .map(
                                                            (
                                                              item
                                                            ) => (
                                                              <article
                                                                key={
                                                                  item.class_label
                                                                }
                                                              >
                                                                <strong>
                                                                  {
                                                                    item.class_label
                                                                  }
                                                                </strong>

                                                                <span>
                                                                  F1 {
                                                                    formatMetric(
                                                                      item.f1
                                                                    )
                                                                  }
                                                                </span>

                                                                <span>
                                                                  Precision {
                                                                    formatMetric(
                                                                      item.precision
                                                                    )
                                                                  }
                                                                </span>

                                                                <span>
                                                                  Recall {
                                                                    formatMetric(
                                                                      item.recall
                                                                    )
                                                                  }
                                                                </span>

                                                                <small>
                                                                  Support {
                                                                    item.support
                                                                  }
                                                                </small>
                                                              </article>
                                                            )
                                                          )
                                                      }
                                                    </div>
                                                  </section>
                                                )
                                              : null
                                          }


                                          <section
                                            className={
                                              styles.evaluationBlock
                                            }
                                          >
                                            <div
                                              className={
                                                styles.evaluationBlockHeader
                                              }
                                            >
                                              <span>
                                                Importance des variables
                                              </span>

                                              <strong>
                                                Permutation importance
                                              </strong>
                                            </div>


                                            <div
                                              className={
                                                styles.importanceList
                                              }
                                            >
                                              {
                                                selectedEvaluation
                                                  .explainability
                                                  .feature_importances
                                                  .map(
                                                    (
                                                      item
                                                    ) => (
                                                      <div
                                                        key={
                                                          item.feature_name
                                                        }
                                                      >
                                                        <span
                                                          className={
                                                            styles.importanceRank
                                                          }
                                                        >
                                                          {
                                                            item.rank
                                                          }
                                                        </span>

                                                        <strong>
                                                          {
                                                            item.feature_name
                                                          }
                                                        </strong>

                                                        <span>
                                                          {
                                                            formatMetric(
                                                              item.importance_mean
                                                            )
                                                          }
                                                        </span>
                                                      </div>
                                                    )
                                                  )
                                              }
                                            </div>
                                          </section>


                                          <section
                                            className={
                                              styles.evaluationLimitations
                                            }
                                          >
                                            <span>
                                              Limites de l’évaluation
                                            </span>

                                            <ul>
                                              {
                                                selectedEvaluation
                                                  .limitations
                                                  .map(
                                                    (
                                                      limitation
                                                    ) => (
                                                      <li
                                                        key={
                                                          limitation
                                                        }
                                                      >
                                                        {
                                                          evaluationLimitationLabel(
                                                            limitation
                                                          )
                                                        }
                                                      </li>
                                                    )
                                                  )
                                              }
                                            </ul>
                                          </section>
                                        </section>
                                      )
                                    : null
                                }


                                {
                                  detailLoading
                                    ? (
                                        <div
                                          className={
                                            styles.detailState
                                          }
                                        >
                                          <span
                                            className={
                                              styles.loadingDot
                                            }
                                            aria-hidden="true"
                                          />

                                          <span>
                                            Chargement de la fiche détaillée…
                                          </span>
                                        </div>
                                      )
                                    : null
                                }


                                {
                                  detailError
                                    ? (
                                        <div
                                          className={
                                            `${styles.detailState} ${styles.errorPanel}`
                                          }
                                        >
                                          {
                                            detailError
                                          }
                                        </div>
                                      )
                                    : null
                                }


                                {
                                  selectedDetail
                                    ? (
                                        <div
                                          className={
                                            styles.detailSections
                                          }
                                        >
                                          <section
                                            className={
                                              styles.detailBlock
                                            }
                                          >
                                            <div
                                              className={
                                                styles.detailBlockHeader
                                              }
                                            >
                                              <span>
                                                Données
                                              </span>

                                              <strong>
                                                {
                                                  selectedDetail.train_rows +
                                                  selectedDetail.test_rows
                                                } {" "}
                                                {
                                                  selectedDetail.split.strategy ===
                                                    "purged_group_time_holdout"
                                                    ? "lignes utilisées"
                                                    : "lignes"
                                                }
                                              </strong>
                                            </div>


                                            <dl
                                              className={
                                                styles.definitionGrid
                                              }
                                            >
                                              <div>
                                                <dt>
                                                  Entraînement
                                                </dt>

                                                <dd>
                                                  {
                                                    selectedDetail.train_rows
                                                  }
                                                </dd>
                                              </div>

                                              <div>
                                                <dt>
                                                  Test
                                                </dt>

                                                <dd>
                                                  {
                                                    selectedDetail.test_rows
                                                  }
                                                </dd>
                                              </div>

                                              <div>
                                                <dt>
                                                  Dataset
                                                </dt>

                                                <dd
                                                  title={
                                                    selectedDetail.dataset_id
                                                  }
                                                >
                                                  {
                                                    shortIdentifier(
                                                      selectedDetail.dataset_id
                                                    )
                                                  }
                                                </dd>
                                              </div>

                                              <div>
                                                <dt>
                                                  Features
                                                </dt>

                                                <dd>
                                                  {
                                                    selectedDetail.feature_columns.length
                                                  }
                                                </dd>
                                              </div>
                                            </dl>
                                          </section>


                                          <section
                                            className={
                                              styles.detailBlock
                                            }
                                          >
                                            <div
                                              className={
                                                styles.detailBlockHeader
                                              }
                                            >
                                              <span>
                                                Variables
                                              </span>

                                              <strong>
                                                {
                                                  selectedDetail.categorical_feature_columns.length
                                                } catégorielle(s)
                                              </strong>
                                            </div>


                                            <div
                                              className={
                                                styles.featureList
                                              }
                                            >
                                              {
                                                selectedDetail.feature_columns.map(
                                                  (
                                                    feature
                                                  ) => {
                                                    const categorical =
                                                      selectedDetail
                                                        .categorical_feature_columns
                                                        .includes(
                                                          feature
                                                        );


                                                    return (
                                                      <span
                                                        key={
                                                          feature
                                                        }
                                                        className={
                                                          categorical
                                                            ? styles.featureCategorical
                                                            : styles.featureNumeric
                                                        }
                                                      >
                                                        {
                                                          feature
                                                        }

                                                        <small>
                                                          {
                                                            categorical
                                                              ? "cat."
                                                              : "num."
                                                          }
                                                        </small>
                                                      </span>
                                                    );
                                                  }
                                                )
                                              }
                                            </div>
                                          </section>


                                          <section
                                            className={
                                              styles.detailBlock
                                            }
                                          >
                                            <div
                                              className={
                                                styles.detailBlockHeader
                                              }
                                            >
                                              <span>
                                                Prétraitement
                                              </span>

                                              <strong>
                                                Leakage-safe
                                              </strong>
                                            </div>


                                            <dl
                                              className={
                                                styles.definitionGrid
                                              }
                                            >
                                              <div>
                                                <dt>
                                                  Imputation numérique
                                                </dt>

                                                <dd>
                                                  {
                                                    selectedDetail.preprocessing.numeric_imputation
                                                  }
                                                </dd>
                                              </div>

                                              <div>
                                                <dt>
                                                  Imputation catégorielle
                                                </dt>

                                                <dd>
                                                  {
                                                    selectedDetail.preprocessing.categorical_imputation
                                                  }
                                                </dd>
                                              </div>

                                              <div>
                                                <dt>
                                                  Encodage
                                                </dt>

                                                <dd>
                                                  {
                                                    selectedDetail.preprocessing.categorical_encoding
                                                  }
                                                </dd>
                                              </div>

                                              <div>
                                                <dt>
                                                  Standardisation
                                                </dt>

                                                <dd>
                                                  {
                                                    selectedDetail.preprocessing.scale_numeric
                                                      ? "Oui"
                                                      : "Non"
                                                  }
                                                </dd>
                                              </div>
                                            </dl>
                                          </section>


                                          <section
                                            className={
                                              styles.detailBlock
                                            }
                                          >
                                            <div
                                              className={
                                                styles.detailBlockHeader
                                              }
                                            >
                                              <span>
                                                Split
                                              </span>

                                              <strong>
                                                {
                                                  selectedDetail.split.strategy ===
                                                    "purged_group_time_holdout"
                                                    ? "Chronologique + purge entités"
                                                    : (
                                                        selectedDetail.split.strategy ===
                                                          "group_holdout"
                                                          ? "Par entité"
                                                          : (
                                                              selectedDetail.split.strategy ===
                                                                "time_holdout"
                                                                ? "Chronologique"
                                                                : "Par lignes"
                                                            )
                                                      )
                                                }
                                              </strong>
                                            </div>


                                            <dl
                                              className={
                                                styles.definitionGrid
                                              }
                                            >
                                              <div>
                                                <dt>
                                                  Test size
                                                </dt>

                                                <dd>
                                                  {
                                                    formatMetric(
                                                      selectedDetail.split.test_size
                                                    )
                                                  }
                                                </dd>
                                              </div>

                                              {
                                                (
                                                  selectedDetail.split.strategy ===
                                                    "group_holdout"
                                                  ||
                                                  selectedDetail.split.strategy ===
                                                    "purged_group_time_holdout"
                                                )
                                                  ? (
                                                      <div>
                                                        <dt>
                                                          Entité
                                                        </dt>

                                                        <dd>
                                                          {
                                                            selectedDetail
                                                              .split
                                                              .group_column
                                                          }
                                                        </dd>
                                                      </div>
                                                    )
                                                  : null
                                              }

                                              {
                                                (
                                                  selectedDetail.split.strategy ===
                                                    "time_holdout"
                                                  ||
                                                  selectedDetail.split.strategy ===
                                                    "purged_group_time_holdout"
                                                )
                                                  ? (
                                                      <div>
                                                        <dt>
                                                          Temps
                                                        </dt>

                                                        <dd>
                                                          {
                                                            selectedDetail
                                                              .split
                                                              .time_column
                                                          }
                                                        </dd>
                                                      </div>
                                                    )
                                                  : null
                                              }

                                              {
                                                selectedDetail.split.strategy ===
                                                  "purged_group_time_holdout"
                                                  ? (
                                                      <div>
                                                        <dt>
                                                          Politique
                                                        </dt>

                                                        <dd>
                                                          Entités futures purgées du train
                                                        </dd>
                                                      </div>
                                                    )
                                                  : null
                                              }

                                              <div>
                                                <dt>
                                                  Seed
                                                </dt>

                                                <dd>
                                                  {
                                                    selectedDetail.split.random_seed
                                                  }
                                                </dd>
                                              </div>

                                              <div>
                                                <dt>
                                                  Shuffle
                                                </dt>

                                                <dd>
                                                  {
                                                    selectedDetail.split.shuffle
                                                      ? "Oui"
                                                      : "Non"
                                                  }
                                                </dd>
                                              </div>

                                              <div>
                                                <dt>
                                                  Stratification
                                                </dt>

                                                <dd>
                                                  {
                                                    selectedDetail.split.stratify
                                                      ? "Oui"
                                                      : "Non"
                                                  }
                                                </dd>
                                              </div>
                                            </dl>
                                          </section>


                                          <section
                                            className={
                                              styles.detailBlock
                                            }
                                          >
                                            <div
                                              className={
                                                styles.detailBlockHeader
                                              }
                                            >
                                              <span>
                                                Hyperparamêtres
                                              </span>

                                              <strong>
                                                {
                                                  selectedDetail
                                                    .effective_estimator_hyperparameters
                                                    .kind
                                                }
                                              </strong>
                                            </div>


                                            <dl
                                              className={
                                                styles.definitionGrid
                                              }
                                            >
                                              {
                                                hyperparameterEntries(
                                                  selectedDetail
                                                    .effective_estimator_hyperparameters
                                                ).map(
                                                  (
                                                    [
                                                      key,
                                                      value,
                                                    ]
                                                  ) => (
                                                    <div
                                                      key={
                                                        key
                                                      }
                                                    >
                                                      <dt>
                                                        {
                                                          key
                                                        }
                                                      </dt>

                                                      <dd>
                                                        {
                                                          value
                                                        }
                                                      </dd>
                                                    </div>
                                                  )
                                                )
                                              }
                                            </dl>
                                          </section>


                                          <section
                                            className={
                                              styles.detailBlock
                                            }
                                          >
                                            <div
                                              className={
                                                styles.detailBlockHeader
                                              }
                                            >
                                              <span>
                                                Provenance
                                              </span>

                                              <strong>
                                                {
                                                  selectedDetail.has_experiment_provenance
                                                    ? "Vérifiable"
                                                    : "Non disponible"
                                                }
                                              </strong>
                                            </div>


                                            <dl
                                              className={
                                                styles.provenanceGrid
                                              }
                                            >
                                              <div>
                                                <dt>
                                                  Experiment
                                                </dt>

                                                <dd
                                                  title={
                                                    selectedDetail.experiment_id ??
                                                    undefined
                                                  }
                                                >
                                                  {
                                                    selectedDetail.experiment_id
                                                      ? shortIdentifier(
                                                          selectedDetail.experiment_id
                                                        )
                                                      : "?"
                                                  }
                                                </dd>
                                              </div>

                                              <div>
                                                <dt>
                                                  Preparation revision
                                                </dt>

                                                <dd>
                                                  {
                                                    selectedDetail.preparation_session_revision ??
                                                    "?"
                                                  }
                                                </dd>
                                              </div>

                                              <div>
                                                <dt>
                                                  Training contract
                                                </dt>

                                                <dd
                                                  title={
                                                    selectedDetail.training_contract_sha256 ??
                                                    undefined
                                                  }
                                                >
                                                  {
                                                    selectedDetail.training_contract_sha256
                                                      ? shortIdentifier(
                                                          selectedDetail.training_contract_sha256
                                                        )
                                                      : "?"
                                                  }
                                                </dd>
                                              </div>
                                            </dl>
                                          </section>
                                        </div>
                                      )
                                    : null
                                }
                              </>
                            )
                          : null
                      }
                    </section>
                  </div>
                )
              : null
          }
        </section>
      </div>
    </main>
  );
}
