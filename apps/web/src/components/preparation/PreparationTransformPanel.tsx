"use client";

import {
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  getPreparationAnalysisOutputCandidates,
  getPreparationSession,
} from "./preparationApi";

import {
  applyPreparationTransformation,
  buildPreparationTransformationPlan,
} from "./transformationApi";

import type {
  PreparationAnalysisOutputCandidate,
  PreparationSessionView,
  PreparationStageRecord,
} from "./preparationTypes";

import type {
  PreparationTransformationApprovalCommand,
  PreparationTransformationIntent,
  PreparationTransformationPlan,
  TransformationAggregationFunction,
  TransformationArithmeticOperator,
  TransformationCastTargetType,
  TransformationDatePart,
  TransformationOperandKind,
  TransformationOperation,
} from "./transformationTypes";

import styles from "./PreparationTransformPanel.module.css";


type PreparationTransformPanelProps = {
  session:
    PreparationSessionView |
    null;

  onSessionChange?:
    (
      session:
        PreparationSessionView
    ) => void;
};


type ApprovalChoice =
  | "approve"
  | "reject"
  | "defer";


type ArithmeticDraft = {
  outputColumn:
    string;

  leftKind:
    TransformationOperandKind;

  leftValue:
    string;

  operator:
    TransformationArithmeticOperator;

  rightKind:
    TransformationOperandKind;

  rightValue:
    string;
};


type CastDraft = {
  sourceColumn:
    string;

  outputColumn:
    string;

  targetType:
    TransformationCastTargetType;
};


type BinDraft = {
  sourceColumn:
    string;

  outputColumn:
    string;

  bins:
    string;

  labels:
    string;
};


type DateDraft = {
  sourceColumn:
    string;

  outputColumn:
    string;

  part:
    TransformationDatePart;
};


type AggregateDraft = {
  groupBy:
    string;

  sourceColumn:
    string;

  function:
    TransformationAggregationFunction;

  outputColumn:
    string;

  outputFilename:
    string;
};


function findStage(
  session:
    PreparationSessionView,

  stageName:
    PreparationStageRecord[
      "stage"
    ]
): PreparationStageRecord | null {
  return (
    session
      .snapshot
      .stages
      .find(
        (
          stage
        ) =>
          stage.stage ===
          stageName
      ) ??
    null
  );
}


function terminalCandidateDatasetIds(
  candidates:
    PreparationAnalysisOutputCandidate[]
): Set<string> {
  const usedAsParent =
    new Set(
      candidates.flatMap(
        (
          candidate
        ) =>
          candidate
            .parent_dataset_ids
            .filter(
              (
                parentDatasetId
              ) =>
                parentDatasetId !==
                candidate.dataset_id
            )
      )
    );


  return new Set(
    candidates
      .filter(
        (
          candidate
        ) =>
          !usedAsParent.has(
            candidate.dataset_id
          )
      )
      .map(
        (
          candidate
        ) =>
          candidate.dataset_id
      )
  );
}


function operationLabel(
  operation:
    TransformationOperation
): string {
  switch (
    operation
  ) {
    case "derive_arithmetic":
      return "Créer une variable";

    case "cast":
      return "Convertir un type";

    case "bin_numeric":
      return "Créer des classes";

    case "extract_date_part":
      return "Extraire une date";

    case "aggregate":
      return "Agréger";

    default:
      return operation;
  }
}


function riskLabel(
  risk:
    string
): string {
  switch (
    risk
  ) {
    case "low":
      return "Faible";

    case "medium":
      return "Modéré";

    case "high":
      return "Élevé";

    default:
      return risk;
  }
}


function plannerStatusLabel(
  status:
    string
): string {
  switch (
    status
  ) {
    case "validated":
      return "Validée";

    case "review_required":
      return "Approbation requise";

    default:
      return status;
  }
}


function artifactStageLabel(
  stage:
    string
): string {
  switch (
    stage
  ) {
    case "source":
      return "Source";

    case "clean":
      return "Nettoyé";

    case "transform":
      return "Transformé";

    case "combine":
      return "Combiné";

    default:
      return stage;
  }
}


function newRequestId(
  operation:
    TransformationOperation
): string {
  const suffix =
    typeof crypto !==
      "undefined" &&
    typeof crypto.randomUUID ===
      "function"
      ? crypto.randomUUID()
      : `${
          Date.now()
        }-${
          Math.random()
            .toString(
              36
            )
            .slice(
              2
            )
        }`;


  return (
    `transform-ui:${operation}:${suffix}`
  );
}


function parseNumericLiteral(
  value:
    string,

  fieldName:
    string
): number {
  const normalized =
    value
      .trim()
      .replace(
        ",",
        "."
      );


  const parsed =
    Number(
      normalized
    );


  if (
    !normalized ||
    !Number.isFinite(
      parsed
    )
  ) {
    throw new Error(
      `${fieldName} doit être un nombre valide.`
    );
  }


  return parsed;
}


function operandFromDraft(
  kind:
    TransformationOperandKind,

  raw:
    string,

  fieldName:
    string
) {
  const normalized =
    raw.trim();


  if (
    !normalized
  ) {
    throw new Error(
      `${fieldName} est requis.`
    );
  }


  if (
    kind ===
    "column"
  ) {
    return {
      kind:
        "column" as const,

      column:
        normalized,

      value:
        null,
    };
  }


  return {
    kind:
      "literal" as const,

    column:
      null,

    value:
      parseNumericLiteral(
        normalized,
        fieldName
      ),
  };
}


function commaSeparatedValues(
  value:
    string
): string[] {
  return value
    .split(
      ","
    )
    .map(
      (
        item
      ) =>
        item.trim()
    )
    .filter(
      Boolean
    );
}


function transformationSummary(
  intent:
    PreparationTransformationIntent
): string {
  switch (
    intent.operation
  ) {
    case "derive_arithmetic": {
      const left =
        intent.left.kind ===
          "column"
          ? intent.left.column
          : String(
              intent.left.value
            );

      const right =
        intent.right.kind ===
          "column"
          ? intent.right.column
          : String(
              intent.right.value
            );


      return (
        `${intent.output_column} = ${
          left
        } ${
          intent.operator
        } ${
          right
        }`
      );
    }


    case "cast":
      return (
        `${intent.source_column} → ${
          intent.output_column
        } · ${
          intent.target_type
        }`
      );


    case "bin_numeric":
      return (
        `${intent.source_column} → ${
          intent.output_column
        } · ${
          intent.bins.length
        } bornes`
      );


    case "extract_date_part":
      return (
        `${intent.source_column} → ${
          intent.output_column
        } · ${
          intent.part
        }`
      );


    case "aggregate":
      return (
        `${
          intent.group_by.join(
            " + "
          )
        } · ${
          intent.metrics[
            0
          ]?.function ??
          "agrégation"
        }`
      );


    default:
      return "Transformation";
  }
}


export default function PreparationTransformPanel({
  session,
  onSessionChange,
}: PreparationTransformPanelProps) {
  const [
    candidates,
    setCandidates,
  ] = useState<
    PreparationAnalysisOutputCandidate[]
  >(
    []
  );


  const [
    candidatesLoading,
    setCandidatesLoading,
  ] = useState(
    false
  );


  const [
    candidatesError,
    setCandidatesError,
  ] = useState<
    string |
    null
  >(
    null
  );


  const [
    selectedDatasetId,
    setSelectedDatasetId,
  ] = useState(
    ""
  );


  const [
    operation,
    setOperation,
  ] = useState<
    TransformationOperation
  >(
    "derive_arithmetic"
  );


  const [
    arithmeticDraft,
    setArithmeticDraft,
  ] = useState<
    ArithmeticDraft
  >({
    outputColumn:
      "",

    leftKind:
      "column",

    leftValue:
      "",

    operator:
      "multiply",

    rightKind:
      "column",

    rightValue:
      "",
  });


  const [
    castDraft,
    setCastDraft,
  ] = useState<
    CastDraft
  >({
    sourceColumn:
      "",

    outputColumn:
      "",

    targetType:
      "float",
  });


  const [
    binDraft,
    setBinDraft,
  ] = useState<
    BinDraft
  >({
    sourceColumn:
      "",

    outputColumn:
      "",

    bins:
      "",

    labels:
      "",
  });


  const [
    dateDraft,
    setDateDraft,
  ] = useState<
    DateDraft
  >({
    sourceColumn:
      "",

    outputColumn:
      "",

    part:
      "month",
  });


  const [
    aggregateDraft,
    setAggregateDraft,
  ] = useState<
    AggregateDraft
  >({
    groupBy:
      "",

    sourceColumn:
      "",

    function:
      "sum",

    outputColumn:
      "",

    outputFilename:
      "",
  });


  const [
    intents,
    setIntents,
  ] = useState<
    PreparationTransformationIntent[]
  >(
    []
  );


  const [
    plan,
    setPlan,
  ] = useState<
    PreparationTransformationPlan |
    null
  >(
    null
  );


  const [
    approvals,
    setApprovals,
  ] = useState<
    Record<
      string,
      ApprovalChoice
    >
  >(
    {}
  );


  const [
    planning,
    setPlanning,
  ] = useState(
    false
  );


  const [
    applying,
    setApplying,
  ] = useState(
    false
  );


  const [
    draftError,
    setDraftError,
  ] = useState<
    string |
    null
  >(
    null
  );


  const [
    workflowError,
    setWorkflowError,
  ] = useState<
    string |
    null
  >(
    null
  );


  const transform =
    session !==
      null
      ? findStage(
          session,
          "transform"
        )
      : null;


  const transformResolved =
    transform?.status ===
      "passed" ||
    transform?.status ===
      "skipped";


  useEffect(
    () => {
      setCandidates(
        []
      );

      setSelectedDatasetId(
        ""
      );

      setIntents(
        []
      );

      setPlan(
        null
      );

      setApprovals(
        {}
      );

      setDraftError(
        null
      );

      setWorkflowError(
        null
      );
    },
    [
      session?.workflow_id,
    ]
  );


  useEffect(
    () => {
      const currentSession =
        session;


      if (
        currentSession ===
          null ||
        transformResolved
      ) {
        return;
      }


      /*
       * On extrait immédiatement workflow_id.
       *
       * La valeur devient une string non nullable et peut être
       * utilisée sans ambiguïté à l'intérieur de la fonction async.
       */
      const workflowId =
        currentSession.workflow_id;


      const controller =
        new AbortController();


      async function loadCandidates() {
        setCandidatesLoading(
          true
        );

        setCandidatesError(
          null
        );


        try {
          const response =
            await getPreparationAnalysisOutputCandidates(
              workflowId,
              controller.signal
            );


          const terminalIds =
            terminalCandidateDatasetIds(
              response.candidates
            );


          const activeCandidates =
            response
              .candidates
              .filter(
                (
                  candidate
                ) =>
                  terminalIds.has(
                    candidate.dataset_id
                  ) &&
                  candidate.stage !==
                    "combine"
              );


          setCandidates(
            activeCandidates
          );


          setSelectedDatasetId(
            (
              current
            ) => {
              if (
                activeCandidates.some(
                  (
                    candidate
                  ) =>
                    candidate.dataset_id ===
                    current
                )
              ) {
                return current;
              }


              return (
                activeCandidates[
                  0
                ]?.dataset_id ??
                ""
              );
            }
          );
        } catch (
          caughtError
        ) {
          if (
            controller.signal.aborted
          ) {
            return;
          }


          setCandidatesError(
            caughtError
              instanceof Error
              ? caughtError.message
              : "Impossible de charger les datasets transformables."
          );
        } finally {
          if (
            !controller.signal.aborted
          ) {
            setCandidatesLoading(
              false
            );
          }
        }
      }


      void loadCandidates();


      return () => {
        controller.abort();
      };
    },
    [
      session,
      transformResolved,
    ]
  );


  const selectedDataset =
    useMemo(
      () =>
        candidates.find(
          (
            candidate
          ) =>
            candidate.dataset_id ===
            selectedDatasetId
        ) ??
        null,
      [
        candidates,
        selectedDatasetId,
      ]
    );


  const reviewRequiredSteps =
    plan
      ?.steps
      .filter(
        (
          step
        ) =>
          step.requires_human_approval ||
          step.status ===
            "review_required"
      ) ??
    [];


  const unresolvedApprovalCount =
    reviewRequiredSteps.filter(
      (
        step
      ) =>
        !approvals[
          step.request_id
        ]
    ).length;


  const canApplyPlan =
    plan !==
      null &&
    plan.ready_for_approval &&
    unresolvedApprovalCount ===
      0 &&
    !applying;


  /*
   * Tous les hooks React ont maintenant été exécutés.
   *
   * À partir d'ici, les retours conditionnels sont sûrs.
   */
  const activeSession =
    session;


  if (
    activeSession ===
    null
  ) {
    return null;
  }


  /*
   * À partir de ce point, workflowId est définitivement
   * une string non nullable.
   *
   * Les handlers async n'accèdent donc plus directement
   * à session.workflow_id.
   */
  const workflowId =
    activeSession.workflow_id;


  function synchronizeSession(
    nextSession:
      PreparationSessionView
  ) {
    onSessionChange?.(
      nextSession
    );
  }


  function clearPlan() {
    setPlan(
      null
    );

    setApprovals(
      {}
    );

    setWorkflowError(
      null
    );
  }


  function resetOperationForm() {
    setArithmeticDraft(
      {
        outputColumn:
          "",

        leftKind:
          "column",

        leftValue:
          "",

        operator:
          "multiply",

        rightKind:
          "column",

        rightValue:
          "",
      }
    );


    setCastDraft(
      {
        sourceColumn:
          "",

        outputColumn:
          "",

        targetType:
          "float",
      }
    );


    setBinDraft(
      {
        sourceColumn:
          "",

        outputColumn:
          "",

        bins:
          "",

        labels:
          "",
      }
    );


    setDateDraft(
      {
        sourceColumn:
          "",

        outputColumn:
          "",

        part:
          "month",
      }
    );


    setAggregateDraft(
      {
        groupBy:
          "",

        sourceColumn:
          "",

        function:
          "sum",

        outputColumn:
          "",

        outputFilename:
          "",
      }
    );
  }


  function handleAddTransformation() {
    setDraftError(
      null
    );


    if (
      selectedDataset ===
      null
    ) {
      setDraftError(
        "Sélectionnez d’abord un dataset."
      );

      return;
    }


    try {
      let intent:
        PreparationTransformationIntent;


      switch (
        operation
      ) {
        case "derive_arithmetic": {
          const outputColumn =
            arithmeticDraft
              .outputColumn
              .trim();


          if (
            !outputColumn
          ) {
            throw new Error(
              "La colonne de sortie est requise."
            );
          }


          intent = {
            request_id:
              newRequestId(
                operation
              ),

            dataset_id:
              selectedDataset
                .dataset_id,

            dataset_filename:
              selectedDataset
                .dataset_filename,

            operation:
              "derive_arithmetic",

            output_column:
              outputColumn,

            left:
              operandFromDraft(
                arithmeticDraft
                  .leftKind,

                arithmeticDraft
                  .leftValue,

                "L’opérande gauche"
              ),

            operator:
              arithmeticDraft
                .operator,

            right:
              operandFromDraft(
                arithmeticDraft
                  .rightKind,

                arithmeticDraft
                  .rightValue,

                "L’opérande droit"
              ),
          };

          break;
        }


        case "cast": {
          const sourceColumn =
            castDraft
              .sourceColumn
              .trim();

          const outputColumn =
            castDraft
              .outputColumn
              .trim();


          if (
            !sourceColumn ||
            !outputColumn
          ) {
            throw new Error(
              "Les colonnes source et de sortie sont requises."
            );
          }


          intent = {
            request_id:
              newRequestId(
                operation
              ),

            dataset_id:
              selectedDataset
                .dataset_id,

            dataset_filename:
              selectedDataset
                .dataset_filename,

            operation:
              "cast",

            source_column:
              sourceColumn,

            output_column:
              outputColumn,

            target_type:
              castDraft
                .targetType,
          };

          break;
        }


        case "bin_numeric": {
          const sourceColumn =
            binDraft
              .sourceColumn
              .trim();

          const outputColumn =
            binDraft
              .outputColumn
              .trim();

          const rawBins =
            commaSeparatedValues(
              binDraft.bins
            );


          const bins =
            rawBins.map(
              (
                item
              ) =>
                parseNumericLiteral(
                  item,
                  "Chaque borne"
                )
            );


          if (
            !sourceColumn ||
            !outputColumn
          ) {
            throw new Error(
              "Les colonnes source et de sortie sont requises."
            );
          }


          if (
            bins.length <
            2
          ) {
            throw new Error(
              "Indiquez au moins deux bornes numériques."
            );
          }


          const labels =
            commaSeparatedValues(
              binDraft.labels
            );


          intent = {
            request_id:
              newRequestId(
                operation
              ),

            dataset_id:
              selectedDataset
                .dataset_id,

            dataset_filename:
              selectedDataset
                .dataset_filename,

            operation:
              "bin_numeric",

            source_column:
              sourceColumn,

            output_column:
              outputColumn,

            bins,

            labels:
              labels.length >
                0
                ? labels
                : null,

            include_lowest:
              true,

            right:
              true,
          };

          break;
        }


        case "extract_date_part": {
          const sourceColumn =
            dateDraft
              .sourceColumn
              .trim();

          const outputColumn =
            dateDraft
              .outputColumn
              .trim();


          if (
            !sourceColumn ||
            !outputColumn
          ) {
            throw new Error(
              "Les colonnes source et de sortie sont requises."
            );
          }


          intent = {
            request_id:
              newRequestId(
                operation
              ),

            dataset_id:
              selectedDataset
                .dataset_id,

            dataset_filename:
              selectedDataset
                .dataset_filename,

            operation:
              "extract_date_part",

            source_column:
              sourceColumn,

            output_column:
              outputColumn,

            part:
              dateDraft.part,
          };

          break;
        }


        case "aggregate": {
          const groupBy =
            commaSeparatedValues(
              aggregateDraft
                .groupBy
            );

          const sourceColumn =
            aggregateDraft
              .sourceColumn
              .trim();

          const outputColumn =
            aggregateDraft
              .outputColumn
              .trim();


          if (
            groupBy.length ===
              0
          ) {
            throw new Error(
              "Indiquez au moins une colonne de regroupement."
            );
          }


          if (
            !sourceColumn ||
            !outputColumn
          ) {
            throw new Error(
              "La mesure source et sa colonne de sortie sont requises."
            );
          }


          const outputFilename =
            aggregateDraft
              .outputFilename
              .trim() ||
            `${
              selectedDataset
                .dataset_filename
                .replace(
                  /\.[^.]+$/,
                  ""
                )
            }__aggregate.csv`;


          const outputDatasetId =
            `${
              selectedDataset.dataset_id
            }:aggregate:${
              Date.now()
            }`;


          intent = {
            request_id:
              newRequestId(
                operation
              ),

            dataset_id:
              selectedDataset
                .dataset_id,

            dataset_filename:
              selectedDataset
                .dataset_filename,

            operation:
              "aggregate",

            group_by:
              groupBy,

            metrics: [
              {
                source_column:
                  sourceColumn,

                function:
                  aggregateDraft
                    .function,

                output_column:
                  outputColumn,
              },
            ],

            output_dataset_id:
              outputDatasetId,

            output_dataset_filename:
              outputFilename,
          };

          break;
        }
      }


      setIntents(
        (
          current
        ) => [
          ...current,
          intent,
        ]
      );


      clearPlan();

      resetOperationForm();
    } catch (
      caughtError
    ) {
      setDraftError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "La transformation n’est pas valide."
      );
    }
  }


  function handleRemoveIntent(
    requestId:
      string
  ) {
    setIntents(
      (
        current
      ) =>
        current.filter(
          (
            intent
          ) =>
            intent.request_id !==
            requestId
        )
    );


    clearPlan();
  }


  async function refreshSession() {
    const refreshed =
      await getPreparationSession(
        workflowId
      );


    synchronizeSession(
      refreshed
    );


    return refreshed;
  }


  async function handleBuildPlan() {
    if (
      !selectedDatasetId ||
      intents.length ===
        0
    ) {
      return;
    }


    setPlanning(
      true
    );

    setWorkflowError(
      null
    );

    setDraftError(
      null
    );


    try {
      const response =
        await buildPreparationTransformationPlan(
          workflowId,
          selectedDatasetId,
          intents
        );


      setPlan(
        response
      );


      setApprovals(
        {}
      );


      await refreshSession();
    } catch (
      caughtError
    ) {
      setWorkflowError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "La construction du plan de transformation a échoué."
      );
    } finally {
      setPlanning(
        false
      );
    }
  }


  async function handleApplyPlan() {
    if (
      !selectedDatasetId ||
      plan ===
        null ||
      !canApplyPlan
    ) {
      return;
    }


    const approvalCommands:
      PreparationTransformationApprovalCommand[] =
        reviewRequiredSteps.map(
          (
            step
          ) => ({
            request_id:
              step.request_id,

            decision:
              approvals[
                step.request_id
              ],

            actor:
              "user",

            comment:
              "Décision enregistrée depuis l’interface Preparation de DataLens.",
          })
        );


    setApplying(
      true
    );

    setWorkflowError(
      null
    );


    try {
      const response =
        await applyPreparationTransformation(
          workflowId,
          selectedDatasetId,
          intents,
          approvalCommands
        );


      if (
        response.status ===
          "validation_failed"
      ) {
        setWorkflowError(
          "La post-validation a rejeté le résultat. Aucun artifact transformé n’a été matérialisé."
        );

        return;
      }


      await refreshSession();
    } catch (
      caughtError
    ) {
      setWorkflowError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "L’exécution des transformations a échoué."
      );
    } finally {
      setApplying(
        false
      );
    }
  }


  async function handleSkipTransform() {
    if (
      !selectedDatasetId
    ) {
      return;
    }


    setApplying(
      true
    );

    setWorkflowError(
      null
    );


    try {
      await applyPreparationTransformation(
        workflowId,
        selectedDatasetId,
        [],
        []
      );


      await refreshSession();
    } catch (
      caughtError
    ) {
      setWorkflowError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "Impossible de terminer l’étape sans transformation."
      );
    } finally {
      setApplying(
        false
      );
    }
  }


  return (
    <section
      className={
        styles.panel
      }
      aria-labelledby="preparation-transform-title"
    >
      <header
        className={
          styles.header
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Préparation structurelle
          </span>

          <h3
            id="preparation-transform-title"
          >
            Transformer les données
          </h3>

          <p>
            Créez uniquement les variables et structures nécessaires
            à l’analyse. DataLens reconstruit et contrôle le plan côté
            serveur avant toute modification des données.
          </p>
        </div>

        <span
          className={
            styles.headerBadge
          }
        >
          PYTHON DÉTERMINISTE
        </span>
      </header>


      <section
        className={
          styles.datasetSection
        }
      >
        <div
          className={
            styles.sectionHeading
          }
        >
          <div>
            <span
              className={
                styles.eyebrow
              }
            >
              01 · Périmètre
            </span>

            <strong>
              Dataset à transformer
            </strong>
          </div>

          {
            candidatesLoading
              ? (
                  <span
                    className={
                      styles.subtleStatus
                    }
                  >
                    Chargement…
                  </span>
                )
              : null
          }
        </div>


        {
          candidatesError
            ? (
                <div
                  className={
                    styles.error
                  }
                  role="alert"
                >
                  {
                    candidatesError
                  }
                </div>
              )
            : null
        }


        {
          candidates.length >
            0
            ? (
                <div
                  className={
                    styles.datasetGrid
                  }
                >
                  {
                    candidates.map(
                      (
                        candidate
                      ) => {
                        const selected =
                          candidate.dataset_id ===
                          selectedDatasetId;


                        return (
                          <button
                            key={
                              candidate.dataset_id
                            }
                            type="button"
                            className={
                              `${styles.datasetCard} ${
                                selected
                                  ? styles.datasetCardSelected
                                  : ""
                              }`
                            }
                            onClick={
                              () => {
                                if (
                                  candidate.dataset_id ===
                                  selectedDatasetId
                                ) {
                                  return;
                                }


                                setSelectedDatasetId(
                                  candidate.dataset_id
                                );

                                setIntents(
                                  []
                                );

                                clearPlan();
                              }
                            }
                          >
                            <span
                              className={
                                styles.datasetStage
                              }
                            >
                              {
                                artifactStageLabel(
                                  candidate.stage
                                )
                              }
                            </span>

                            <strong
                              title={
                                candidate.dataset_filename
                              }
                            >
                              {
                                candidate.dataset_filename
                              }
                            </strong>

                            <span
                              className={
                                styles.datasetMeta
                              }
                            >
                              {
                                candidate.rows
                              }
                              {" lignes · "}
                              {
                                candidate.columns
                              }
                              {" colonnes"}
                            </span>
                          </button>
                        );
                      }
                    )
                  }
                </div>
              )
            : (
                !candidatesLoading
                  ? (
                      <div
                        className={
                          styles.empty
                        }
                      >
                        Aucun artifact transformable n’est disponible.
                      </div>
                    )
                  : null
              )
        }
      </section>


      <section
        className={
          styles.builder
        }
      >
        <div
          className={
            styles.sectionHeading
          }
        >
          <div>
            <span
              className={
                styles.eyebrow
              }
            >
              02 · Transformation
            </span>

            <strong>
              Ajouter une opération
            </strong>
          </div>
        </div>


        <div
          className={
            styles.operationPicker
          }
        >
          {
            (
              [
                "derive_arithmetic",
                "cast",
                "bin_numeric",
                "extract_date_part",
                "aggregate",
              ] as TransformationOperation[]
            ).map(
              (
                item
              ) => (
                <button
                  key={
                    item
                  }
                  type="button"
                  className={
                    `${styles.operationButton} ${
                      operation ===
                        item
                        ? styles.operationButtonActive
                        : ""
                    }`
                  }
                  onClick={
                    () =>
                      setOperation(
                        item
                      )
                  }
                >
                  {
                    operationLabel(
                      item
                    )
                  }
                </button>
              )
            )
          }
        </div>


        <div
          className={
            styles.formCard
          }
        >
          {
            operation ===
              "derive_arithmetic"
              ? (
                  <>
                    <div
                      className={
                        styles.formIntro
                      }
                    >
                      <strong>
                        Créer une variable calculée
                      </strong>

                      <p>
                        Exemple : revenue = price × quantity.
                      </p>
                    </div>

                    <div
                      className={
                        styles.formGrid
                      }
                    >
                      <label>
                        <span>
                          Colonne de sortie
                        </span>

                        <input
                          value={
                            arithmeticDraft.outputColumn
                          }
                          onChange={
                            (
                              event
                            ) =>
                              setArithmeticDraft(
                                (
                                  current
                                ) => ({
                                  ...current,

                                  outputColumn:
                                    event.target.value,
                                })
                              )
                          }
                          placeholder="revenue"
                        />
                      </label>


                      <label>
                        <span>
                          Opérande gauche
                        </span>

                        <div
                          className={
                            styles.compoundField
                          }
                        >
                          <select
                            value={
                              arithmeticDraft.leftKind
                            }
                            onChange={
                              (
                                event
                              ) =>
                                setArithmeticDraft(
                                  (
                                    current
                                  ) => ({
                                    ...current,

                                    leftKind:
                                      event.target.value as TransformationOperandKind,
                                  })
                                )
                            }
                          >
                            <option value="column">
                              Colonne
                            </option>

                            <option value="literal">
                              Nombre
                            </option>
                          </select>

                          <input
                            value={
                              arithmeticDraft.leftValue
                            }
                            onChange={
                              (
                                event
                              ) =>
                                setArithmeticDraft(
                                  (
                                    current
                                  ) => ({
                                    ...current,

                                    leftValue:
                                      event.target.value,
                                  })
                                )
                            }
                            placeholder={
                              arithmeticDraft.leftKind ===
                                "column"
                                ? "price"
                                : "1.2"
                            }
                          />
                        </div>
                      </label>


                      <label>
                        <span>
                          Opérateur
                        </span>

                        <select
                          value={
                            arithmeticDraft.operator
                          }
                          onChange={
                            (
                              event
                            ) =>
                              setArithmeticDraft(
                                (
                                  current
                                ) => ({
                                  ...current,

                                  operator:
                                    event.target.value as TransformationArithmeticOperator,
                                })
                              )
                          }
                        >
                          <option value="add">
                            Addition
                          </option>

                          <option value="subtract">
                            Soustraction
                          </option>

                          <option value="multiply">
                            Multiplication
                          </option>

                          <option value="divide">
                            Division
                          </option>
                        </select>
                      </label>


                      <label>
                        <span>
                          Opérande droit
                        </span>

                        <div
                          className={
                            styles.compoundField
                          }
                        >
                          <select
                            value={
                              arithmeticDraft.rightKind
                            }
                            onChange={
                              (
                                event
                              ) =>
                                setArithmeticDraft(
                                  (
                                    current
                                  ) => ({
                                    ...current,

                                    rightKind:
                                      event.target.value as TransformationOperandKind,
                                  })
                                )
                            }
                          >
                            <option value="column">
                              Colonne
                            </option>

                            <option value="literal">
                              Nombre
                            </option>
                          </select>

                          <input
                            value={
                              arithmeticDraft.rightValue
                            }
                            onChange={
                              (
                                event
                              ) =>
                                setArithmeticDraft(
                                  (
                                    current
                                  ) => ({
                                    ...current,

                                    rightValue:
                                      event.target.value,
                                  })
                                )
                            }
                            placeholder={
                              arithmeticDraft.rightKind ===
                                "column"
                                ? "quantity"
                                : "100"
                            }
                          />
                        </div>
                      </label>
                    </div>
                  </>
                )
              : null
          }


          {
            operation ===
              "cast"
              ? (
                  <>
                    <div
                      className={
                        styles.formIntro
                      }
                    >
                      <strong>
                        Convertir un type
                      </strong>

                      <p>
                        La conversion est contrôlée par Python avant exécution.
                      </p>
                    </div>

                    <div
                      className={
                        styles.formGrid
                      }
                    >
                      <label>
                        <span>
                          Colonne source
                        </span>

                        <input
                          value={
                            castDraft.sourceColumn
                          }
                          onChange={
                            (
                              event
                            ) =>
                              setCastDraft(
                                (
                                  current
                                ) => ({
                                  ...current,

                                  sourceColumn:
                                    event.target.value,
                                })
                              )
                          }
                          placeholder="amount"
                        />
                      </label>

                      <label>
                        <span>
                          Colonne de sortie
                        </span>

                        <input
                          value={
                            castDraft.outputColumn
                          }
                          onChange={
                            (
                              event
                            ) =>
                              setCastDraft(
                                (
                                  current
                                ) => ({
                                  ...current,

                                  outputColumn:
                                    event.target.value,
                                })
                              )
                          }
                          placeholder="amount_numeric"
                        />
                      </label>

                      <label>
                        <span>
                          Type cible
                        </span>

                        <select
                          value={
                            castDraft.targetType
                          }
                          onChange={
                            (
                              event
                            ) =>
                              setCastDraft(
                                (
                                  current
                                ) => ({
                                  ...current,

                                  targetType:
                                    event.target.value as TransformationCastTargetType,
                                })
                              )
                          }
                        >
                          <option value="string">
                            Texte
                          </option>

                          <option value="integer">
                            Entier
                          </option>

                          <option value="float">
                            Décimal
                          </option>

                          <option value="boolean">
                            Booléen
                          </option>

                          <option value="datetime">
                            Date / heure
                          </option>
                        </select>
                      </label>
                    </div>
                  </>
                )
              : null
          }


          {
            operation ===
              "bin_numeric"
              ? (
                  <>
                    <div
                      className={
                        styles.formIntro
                      }
                    >
                      <strong>
                        Créer des classes numériques
                      </strong>

                      <p>
                        Exemple : transformer un âge en tranches d’âge.
                      </p>
                    </div>

                    <div
                      className={
                        styles.formGrid
                      }
                    >
                      <label>
                        <span>
                          Colonne source
                        </span>

                        <input
                          value={
                            binDraft.sourceColumn
                          }
                          onChange={
                            (
                              event
                            ) =>
                              setBinDraft(
                                (
                                  current
                                ) => ({
                                  ...current,

                                  sourceColumn:
                                    event.target.value,
                                })
                              )
                          }
                          placeholder="age"
                        />
                      </label>

                      <label>
                        <span>
                          Colonne de sortie
                        </span>

                        <input
                          value={
                            binDraft.outputColumn
                          }
                          onChange={
                            (
                              event
                            ) =>
                              setBinDraft(
                                (
                                  current
                                ) => ({
                                  ...current,

                                  outputColumn:
                                    event.target.value,
                                })
                              )
                          }
                          placeholder="age_band"
                        />
                      </label>

                      <label
                        className={
                          styles.fullWidth
                        }
                      >
                        <span>
                          Bornes séparées par des virgules
                        </span>

                        <input
                          value={
                            binDraft.bins
                          }
                          onChange={
                            (
                              event
                            ) =>
                              setBinDraft(
                                (
                                  current
                                ) => ({
                                  ...current,

                                  bins:
                                    event.target.value,
                                })
                              )
                          }
                          placeholder="0, 18, 30, 45, 65, 120"
                        />
                      </label>

                      <label
                        className={
                          styles.fullWidth
                        }
                      >
                        <span>
                          Libellés facultatifs
                        </span>

                        <input
                          value={
                            binDraft.labels
                          }
                          onChange={
                            (
                              event
                            ) =>
                              setBinDraft(
                                (
                                  current
                                ) => ({
                                  ...current,

                                  labels:
                                    event.target.value,
                                })
                              )
                          }
                          placeholder="0-17, 18-29, 30-44, 45-64, 65+"
                        />
                      </label>
                    </div>
                  </>
                )
              : null
          }


          {
            operation ===
              "extract_date_part"
              ? (
                  <>
                    <div
                      className={
                        styles.formIntro
                      }
                    >
                      <strong>
                        Extraire une composante temporelle
                      </strong>

                      <p>
                        Créez une année, un mois, un trimestre ou une autre
                        composante à partir d’une date.
                      </p>
                    </div>

                    <div
                      className={
                        styles.formGrid
                      }
                    >
                      <label>
                        <span>
                          Colonne date
                        </span>

                        <input
                          value={
                            dateDraft.sourceColumn
                          }
                          onChange={
                            (
                              event
                            ) =>
                              setDateDraft(
                                (
                                  current
                                ) => ({
                                  ...current,

                                  sourceColumn:
                                    event.target.value,
                                })
                              )
                          }
                          placeholder="order_date"
                        />
                      </label>

                      <label>
                        <span>
                          Colonne de sortie
                        </span>

                        <input
                          value={
                            dateDraft.outputColumn
                          }
                          onChange={
                            (
                              event
                            ) =>
                              setDateDraft(
                                (
                                  current
                                ) => ({
                                  ...current,

                                  outputColumn:
                                    event.target.value,
                                })
                              )
                          }
                          placeholder="order_month"
                        />
                      </label>

                      <label>
                        <span>
                          Composante
                        </span>

                        <select
                          value={
                            dateDraft.part
                          }
                          onChange={
                            (
                              event
                            ) =>
                              setDateDraft(
                                (
                                  current
                                ) => ({
                                  ...current,

                                  part:
                                    event.target.value as TransformationDatePart,
                                })
                              )
                          }
                        >
                          <option value="year">
                            Année
                          </option>

                          <option value="month">
                            Mois
                          </option>

                          <option value="day">
                            Jour
                          </option>

                          <option value="quarter">
                            Trimestre
                          </option>

                          <option value="week">
                            Semaine
                          </option>

                          <option value="weekday">
                            Jour de semaine
                          </option>
                        </select>
                      </label>
                    </div>
                  </>
                )
              : null
          }


          {
            operation ===
              "aggregate"
              ? (
                  <>
                    <div
                      className={
                        styles.formIntro
                      }
                    >
                      <strong>
                        Agréger un dataset
                      </strong>

                      <p>
                        Cette opération produit un nouveau dataset dérivé et ne
                        remplace pas silencieusement le grain source.
                      </p>
                    </div>

                    <div
                      className={
                        styles.formGrid
                      }
                    >
                      <label
                        className={
                          styles.fullWidth
                        }
                      >
                        <span>
                          Regrouper par
                        </span>

                        <input
                          value={
                            aggregateDraft.groupBy
                          }
                          onChange={
                            (
                              event
                            ) =>
                              setAggregateDraft(
                                (
                                  current
                                ) => ({
                                  ...current,

                                  groupBy:
                                    event.target.value,
                                })
                              )
                          }
                          placeholder="customer_id, year"
                        />
                      </label>

                      <label>
                        <span>
                          Mesure source
                        </span>

                        <input
                          value={
                            aggregateDraft.sourceColumn
                          }
                          onChange={
                            (
                              event
                            ) =>
                              setAggregateDraft(
                                (
                                  current
                                ) => ({
                                  ...current,

                                  sourceColumn:
                                    event.target.value,
                                })
                              )
                          }
                          placeholder="revenue"
                        />
                      </label>

                      <label>
                        <span>
                          Fonction
                        </span>

                        <select
                          value={
                            aggregateDraft.function
                          }
                          onChange={
                            (
                              event
                            ) =>
                              setAggregateDraft(
                                (
                                  current
                                ) => ({
                                  ...current,

                                  function:
                                    event.target.value as TransformationAggregationFunction,
                                })
                              )
                          }
                        >
                          <option value="sum">
                            Somme
                          </option>

                          <option value="mean">
                            Moyenne
                          </option>

                          <option value="median">
                            Médiane
                          </option>

                          <option value="min">
                            Minimum
                          </option>

                          <option value="max">
                            Maximum
                          </option>

                          <option value="count">
                            Nombre
                          </option>

                          <option value="nunique">
                            Valeurs distinctes
                          </option>
                        </select>
                      </label>

                      <label>
                        <span>
                          Colonne résultante
                        </span>

                        <input
                          value={
                            aggregateDraft.outputColumn
                          }
                          onChange={
                            (
                              event
                            ) =>
                              setAggregateDraft(
                                (
                                  current
                                ) => ({
                                  ...current,

                                  outputColumn:
                                    event.target.value,
                                })
                              )
                          }
                          placeholder="total_revenue"
                        />
                      </label>

                      <label>
                        <span>
                          Nom du dataset dérivé
                        </span>

                        <input
                          value={
                            aggregateDraft.outputFilename
                          }
                          onChange={
                            (
                              event
                            ) =>
                              setAggregateDraft(
                                (
                                  current
                                ) => ({
                                  ...current,

                                  outputFilename:
                                    event.target.value,
                                })
                              )
                          }
                          placeholder="orders_by_customer.csv"
                        />
                      </label>
                    </div>
                  </>
                )
              : null
          }


          {
            draftError
              ? (
                  <div
                    className={
                      styles.error
                    }
                    role="alert"
                  >
                    {
                      draftError
                    }
                  </div>
                )
              : null
          }


          <div
            className={
              styles.formFooter
            }
          >
            <span>
              Les noms de colonnes seront contrôlés sur l’artifact serveur au
              moment de la planification.
            </span>

            <button
              type="button"
              className={
                styles.primaryButton
              }
              onClick={
                handleAddTransformation
              }
              disabled={
                selectedDataset ===
                null
              }
            >
              Ajouter au plan
            </button>
          </div>
        </div>
      </section>


      <section
        className={
          styles.draftSection
        }
      >
        <div
          className={
            styles.sectionHeading
          }
        >
          <div>
            <span
              className={
                styles.eyebrow
              }
            >
              03 · Plan
            </span>

            <strong>
              Transformations demandées
            </strong>
          </div>

          <span
            className={
              styles.counter
            }
          >
            {
              intents.length
            }
          </span>
        </div>


        {
          intents.length >
            0
            ? (
                <div
                  className={
                    styles.intentList
                  }
                >
                  {
                    intents.map(
                      (
                        intent,
                        index
                      ) => (
                        <article
                          key={
                            intent.request_id
                          }
                          className={
                            styles.intentCard
                          }
                        >
                          <span
                            className={
                              styles.intentIndex
                            }
                          >
                            {
                              String(
                                index +
                                1
                              ).padStart(
                                2,
                                "0"
                              )
                            }
                          </span>

                          <div>
                            <strong>
                              {
                                operationLabel(
                                  intent.operation
                                )
                              }
                            </strong>

                            <p>
                              {
                                transformationSummary(
                                  intent
                                )
                              }
                            </p>
                          </div>

                          <button
                            type="button"
                            className={
                              styles.removeButton
                            }
                            onClick={
                              () =>
                                handleRemoveIntent(
                                  intent.request_id
                                )
                            }
                          >
                            Retirer
                          </button>
                        </article>
                      )
                    )
                  }
                </div>
              )
            : (
                <div
                  className={
                    styles.empty
                  }
                >
                  Aucune transformation ajoutée.
                </div>
              )
        }


        <div
          className={
            styles.planActions
          }
        >
          <button
            type="button"
            className={
              styles.secondaryButton
            }
            disabled={
              applying ||
              planning ||
              !selectedDatasetId ||
              intents.length >
                0
            }
            onClick={
              handleSkipTransform
            }
          >
            {
              applying &&
              intents.length ===
                0
                ? "Validation…"
                : "Aucune transformation requise"
            }
          </button>

          <button
            type="button"
            className={
              styles.primaryButton
            }
            disabled={
              planning ||
              applying ||
              intents.length ===
                0 ||
              !selectedDatasetId
            }
            onClick={
              handleBuildPlan
            }
          >
            {
              planning
                ? "Construction du plan…"
                : plan
                  ? "Reconstruire le plan"
                  : "Vérifier le plan"
            }
          </button>
        </div>
      </section>


      {
        plan
          ? (
              <section
                className={
                  styles.reviewSection
                }
              >
                <div
                  className={
                    styles.sectionHeading
                  }
                >
                  <div>
                    <span
                      className={
                        styles.eyebrow
                      }
                    >
                      04 · Contrôle Python
                    </span>

                    <strong>
                      Plan de transformation
                    </strong>
                  </div>

                  <span
                    className={
                      styles.planStatus
                    }
                  >
                    {
                      plan.review_required_count >
                        0
                        ? `${
                            plan.review_required_count
                          } approbation${
                            plan.review_required_count >
                              1
                              ? "s"
                              : ""
                          }`
                        : "Prêt"
                    }
                  </span>
                </div>


                <div
                  className={
                    styles.planMetrics
                  }
                >
                  <div>
                    <span>
                      Étapes
                    </span>

                    <strong>
                      {
                        plan.step_count
                      }
                    </strong>
                  </div>

                  <div>
                    <span>
                      Validées
                    </span>

                    <strong>
                      {
                        plan.validated_count
                      }
                    </strong>
                  </div>

                  <div>
                    <span>
                      Revue
                    </span>

                    <strong>
                      {
                        plan.review_required_count
                      }
                    </strong>
                  </div>
                </div>


                <div
                  className={
                    styles.planStepList
                  }
                >
                  {
                    plan.steps.map(
                      (
                        step,
                        index
                      ) => {
                        const needsApproval =
                          step.requires_human_approval ||
                          step.status ===
                            "review_required";


                        return (
                          <article
                            key={
                              step.step_id
                            }
                            className={
                              `${styles.planStep} ${
                                needsApproval
                                  ? styles.planStepAttention
                                  : styles.planStepValidated
                              }`
                            }
                          >
                            <header>
                              <div>
                                <span>
                                  Étape
                                  {" "}
                                  {
                                    String(
                                      index +
                                      1
                                    ).padStart(
                                      2,
                                      "0"
                                    )
                                  }
                                </span>

                                <strong>
                                  {
                                    operationLabel(
                                      step.operation
                                    )
                                  }
                                </strong>
                              </div>

                              <span
                                className={
                                  styles.stepStatus
                                }
                              >
                                {
                                  plannerStatusLabel(
                                    step.status
                                  )
                                }
                              </span>
                            </header>


                            <div
                              className={
                                styles.stepMeta
                              }
                            >
                              <div>
                                <span>
                                  Risque
                                </span>

                                <strong>
                                  {
                                    riskLabel(
                                      step.risk
                                    )
                                  }
                                </strong>
                              </div>

                              <div>
                                <span>
                                  Entrées
                                </span>

                                <strong>
                                  {
                                    step.input_columns.length >
                                      0
                                      ? step.input_columns.join(
                                          ", "
                                        )
                                      : "—"
                                  }
                                </strong>
                              </div>

                              <div>
                                <span>
                                  Sortie
                                </span>

                                <strong>
                                  {
                                    step.output_column ??
                                    step.output_dataset_filename ??
                                    "—"
                                  }
                                </strong>
                              </div>
                            </div>


                            <p
                              className={
                                styles.rationale
                              }
                            >
                              {
                                step.rationale
                              }
                            </p>


                            {
                              needsApproval
                                ? (
                                    <div
                                      className={
                                        styles.approvalChoices
                                      }
                                    >
                                      <span>
                                        Décision analyste
                                      </span>

                                      <div>
                                        <button
                                          type="button"
                                          className={
                                            approvals[
                                              step.request_id
                                            ] ===
                                              "approve"
                                              ? styles.choiceActive
                                              : ""
                                          }
                                          onClick={
                                            () =>
                                              setApprovals(
                                                (
                                                  current
                                                ) => ({
                                                  ...current,

                                                  [
                                                    step.request_id
                                                  ]:
                                                    "approve",
                                                })
                                              )
                                          }
                                        >
                                          Approuver
                                        </button>

                                        <button
                                          type="button"
                                          className={
                                            approvals[
                                              step.request_id
                                            ] ===
                                              "reject"
                                              ? styles.choiceReject
                                              : ""
                                          }
                                          onClick={
                                            () =>
                                              setApprovals(
                                                (
                                                  current
                                                ) => ({
                                                  ...current,

                                                  [
                                                    step.request_id
                                                  ]:
                                                    "reject",
                                                })
                                              )
                                          }
                                        >
                                          Refuser
                                        </button>

                                        <button
                                          type="button"
                                          className={
                                            approvals[
                                              step.request_id
                                            ] ===
                                              "defer"
                                              ? styles.choiceDefer
                                              : ""
                                          }
                                          onClick={
                                            () =>
                                              setApprovals(
                                                (
                                                  current
                                                ) => ({
                                                  ...current,

                                                  [
                                                    step.request_id
                                                  ]:
                                                    "defer",
                                                })
                                              )
                                          }
                                        >
                                          Reporter
                                        </button>
                                      </div>
                                    </div>
                                  )
                                : (
                                    <div
                                      className={
                                        styles.automaticNotice
                                      }
                                    >
                                      Faible risque · autorisation automatique côté serveur.
                                    </div>
                                  )
                            }
                          </article>
                        );
                      }
                    )
                  }
                </div>


                <div
                  className={
                    styles.executionFooter
                  }
                >
                  <div>
                    <strong>
                      {
                        unresolvedApprovalCount >
                          0
                          ? `${
                              unresolvedApprovalCount
                            } décision${
                              unresolvedApprovalCount >
                                1
                                ? "s"
                                : ""
                            } restante${
                              unresolvedApprovalCount >
                                1
                                ? "s"
                                : ""
                            }`
                          : "Plan prêt à être exécuté"
                      }
                    </strong>

                    <span>
                      Python reconstruira le plan à partir de l’artifact serveur
                      avant l’exécution.
                    </span>
                  </div>

                  <button
                    type="button"
                    className={
                      styles.primaryButton
                    }
                    disabled={
                      !canApplyPlan
                    }
                    onClick={
                      handleApplyPlan
                    }
                  >
                    {
                      applying
                        ? "Transformation en cours…"
                        : "Appliquer les transformations"
                    }
                  </button>
                </div>
              </section>
            )
          : null
      }


      {
        workflowError
          ? (
              <div
                className={
                  styles.error
                }
                role="alert"
              >
                <strong>
                  Transformation impossible
                </strong>

                <span>
                  {
                    workflowError
                  }
                </span>
              </div>
            )
          : null
      }


      <details
        className={
          styles.technicalDetails
        }
      >
        <summary>
          Voir les garanties techniques
        </summary>

        <div>
          <p>
            Le navigateur n’envoie jamais de DataFrame, de plan approuvé,
            de résultat d’exécution ou de résultat de validation.
          </p>

          <p>
            Le serveur recharge l’artifact courant, reconstruit le plan,
            applique les décisions autorisées, exécute avec Python puis
            lance une post-validation indépendante avant toute
            matérialisation.
          </p>
        </div>
      </details>
    </section>
  );
}