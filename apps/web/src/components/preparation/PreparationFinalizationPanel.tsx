"use client";

import {
  useEffect,
  useState,
} from "react";

import {
  explainPreparationAnalysisOutput,
  getPreparationAnalysisOutputCandidates,
  selectPreparationAnalysisOutput,
} from "./preparationApi";

import type {
  PreparationAnalysisOutputCandidate,
  PreparationAnalysisOutputCandidatesResponse,
  PreparationOutputExplanationResponse,
  PreparationSessionView,
  PreparationStageRecord,
  PreparationStageStatus,
} from "./preparationTypes";

import styles from "./PreparationFinalizationPanel.module.css";


/*
 * DATALENS_FINALIZATION_PRODUCT_LANGUAGE_V0_1
 *
 * Business-facing finalization copy describes DataLens product
 * guarantees rather than implementation-specific terminology.
 */


type PreparationFinalizationPanelProps = {
  session:
    PreparationSessionView |
    null;

  loading:
    boolean;

  error:
    string |
    null;

  onValidate:
    () => void;
};


function findStage(
  session:
    PreparationSessionView,

  name:
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
          name
      ) ??
    null
  );
}


function stageStatusLabel(
  status:
    PreparationStageStatus |
    undefined,

  stage:
    "clean" |
    "transform" |
    "combine" |
    "validate",

  materialized:
    boolean = false
): string {
  switch (
    status
  ) {
    case "passed":
      if (
        stage ===
        "clean"
      ) {
        return materialized
          ? "Appliqué"
          : "Validé sans modification";
      }


      if (
        stage ===
        "transform"
      ) {
        return materialized
          ? "Appliquée"
          : "Validé sans transformation";
      }


      if (
        stage ===
        "combine"
      ) {
        return materialized
          ? "Appliquée"
          : "Validé sans combinaison";
      }


      return "Validé";

    case "skipped":
      return "Non requis";

    case "review_required":
      return "À valider";

    case "blocked":
      return "Bloqué";

    case "not_started":
    default:
      return "À faire";
  }
}


function artifactStageLabel(
  candidate:
    PreparationAnalysisOutputCandidate
): string {
  switch (
    candidate.stage
  ) {
    case "combine":
      return "Combiné";

    case "transform":
      return "Transformé";

    case "clean":
      return "Nettoyé";

    case "source":
      return "Source";

    default:
      return candidate.stage;
  }
}


function sameDatasetSelection(
  left:
    string[],

  right:
    string[]
): boolean {
  if (
    left.length !==
    right.length
  ) {
    return false;
  }


  const leftSorted =
    [...left].sort();

  const rightSorted =
    [...right].sort();


  return leftSorted.every(
    (
      datasetId,
      index
    ) =>
      datasetId ===
      rightSorted[
        index
      ]
  );
}


function terminalCandidateDatasetIds(
  candidates:
    PreparationAnalysisOutputCandidate[]
): string[] {
  /*
   * Return the terminal materialized frontier.
   *
   * CLEAN / TRANSFORM may materialize an artifact in-place:
   *
   *   dataset_id = "orders"
   *   parent_dataset_ids = ["orders"]
   *
   * A self-parent is lineage evidence. It does not mean that
   * the current artifact has been superseded.
   */
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


  return candidates
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
    );
}


function collectAncestorDatasetIds(
  datasetId:
    string,

  candidates:
    PreparationAnalysisOutputCandidate[]
): Set<
  string
> {
  const candidateById =
    new Map(
      candidates.map(
        (
          candidate
        ) => [
          candidate.dataset_id,
          candidate,
        ]
      )
    );


  const ancestors =
    new Set<
      string
    >();


  const visit =
    (
      currentId:
        string
    ) => {
      const candidate =
        candidateById.get(
          currentId
        );


      if (
        !candidate
      ) {
        return;
      }


      for (
        const parentId
        of candidate
          .parent_dataset_ids
      ) {
        if (
          parentId ===
            currentId ||
          ancestors.has(
            parentId
          )
        ) {
          continue;
        }


        ancestors.add(
          parentId
        );

        visit(
          parentId
        );
      }
    };


  visit(
    datasetId
  );


  return ancestors;
}


function lineageRelatedDatasetIds(
  datasetId:
    string,

  candidates:
    PreparationAnalysisOutputCandidate[]
): Set<
  string
> {
  const related =
    collectAncestorDatasetIds(
      datasetId,
      candidates
    );


  for (
    const candidate
    of candidates
  ) {
    if (
      candidate.dataset_id ===
      datasetId
    ) {
      continue;
    }


    const candidateAncestors =
      collectAncestorDatasetIds(
        candidate.dataset_id,
        candidates
      );


    if (
      candidateAncestors.has(
        datasetId
      )
    ) {
      related.add(
        candidate.dataset_id
      );
    }
  }


  return related;
}


export default function PreparationFinalizationPanel({
  session,
  loading,
  error,
  onValidate,
}: PreparationFinalizationPanelProps) {
  const [
    localSession,
    setLocalSession,
  ] = useState<
    PreparationSessionView |
    null
  >(
    session
  );


  const [
    candidatesResponse,
    setCandidatesResponse,
  ] = useState<
    PreparationAnalysisOutputCandidatesResponse |
    null
  >(
    null
  );


  const [
    candidateLoading,
    setCandidateLoading,
  ] = useState(
    false
  );


  const [
    candidateError,
    setCandidateError,
  ] = useState<
    string |
    null
  >(
    null
  );


  const [
    selectionLoading,
    setSelectionLoading,
  ] = useState(
    false
  );


  const [
    selectionError,
    setSelectionError,
  ] = useState<
    string |
    null
  >(
    null
  );


  const [
    draftDatasetIds,
    setDraftDatasetIds,
  ] = useState<
    string[]
  >(
    session
      ?.analysis_output_dataset_ids ??
    []
  );


  const [
    outputExplanations,
    setOutputExplanations,
  ] = useState<
    Record<
      string,
      PreparationOutputExplanationResponse
    >
  >(
    {}
  );


  const [
    outputExplanationLoadingIds,
    setOutputExplanationLoadingIds,
  ] = useState<
    string[]
  >(
    []
  );


  useEffect(
    () => {
      setLocalSession(
        session
      );


      setDraftDatasetIds(
        session
          ?.analysis_output_dataset_ids ??
        []
      );


      setOutputExplanations(
        {}
      );


      setOutputExplanationLoadingIds(
        []
      );
    },
    [
      session,
    ]
  );


  useEffect(
    () => {
      const currentSession =
        session;


      if (
        currentSession ===
        null
      ) {
        setCandidatesResponse(
          null
        );

        setCandidateError(
          null
        );

        setCandidateLoading(
          false
        );

        return;
      }


      const workflowId =
        currentSession.workflow_id;


      const controller =
        new AbortController();


      async function loadCandidates() {
        setCandidateLoading(
          true
        );

        setCandidateError(
          null
        );


        try {
          const response =
            await getPreparationAnalysisOutputCandidates(
              workflowId,
              controller.signal
            );


          if (
            controller.signal.aborted
          ) {
            return;
          }


          setCandidatesResponse(
            response
          );


          const committedOutputIds =
            response
              .analysis_output_dataset_ids;


          const terminalIds =
            terminalCandidateDatasetIds(
              response.candidates
            );


          setDraftDatasetIds(
            committedOutputIds.length >
              0
              ? committedOutputIds
              : terminalIds
          );


          setOutputExplanationLoadingIds(
            terminalIds
          );


          for (
            const datasetId
            of terminalIds
          ) {
            void explainPreparationAnalysisOutput(
              workflowId,
              datasetId,
              true,
              controller.signal
            )
              .then(
                (
                  explanationResponse
                ) => {
                  if (
                    controller.signal.aborted
                  ) {
                    return;
                  }


                  setOutputExplanations(
                    (
                      current
                    ) => ({
                      ...current,

                      [
                        datasetId
                      ]:
                        explanationResponse,
                    })
                  );
                }
              )
              .catch(
                (
                  caughtError
                ) => {
                  if (
                    controller.signal.aborted
                  ) {
                    return;
                  }


                  console.error(
                    "Preparation output explanation failed:",
                    caughtError
                  );
                }
              )
              .finally(
                () => {
                  if (
                    controller.signal.aborted
                  ) {
                    return;
                  }


                  setOutputExplanationLoadingIds(
                    (
                      current
                    ) =>
                      current.filter(
                        (
                          currentId
                        ) =>
                          currentId !==
                          datasetId
                      )
                  );
                }
              );
          }
        } catch (
          caughtError
        ) {
          if (
            controller.signal.aborted
          ) {
            return;
          }


          setCandidateError(
            caughtError
              instanceof Error
              ? caughtError.message
              : "Impossible de charger les sorties de préparation disponibles."
          );
        } finally {
          if (
            !controller.signal.aborted
          ) {
            setCandidateLoading(
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
      session
        ?.workflow_id,

      session
        ?.revision,
    ]
  );


  /*
   * Everything below this guard contains no React hooks.
   *
   * This keeps the hook order stable whether a Preparation
   * session exists or not.
   */
  if (
    session ===
    null
  ) {
    return null;
  }


  const effectiveSession =
    localSession ??
    session;


  const snapshot =
    effectiveSession.snapshot;


  const clean =
    findStage(
      effectiveSession,
      "clean"
    );


  const transform =
    findStage(
      effectiveSession,
      "transform"
    );


  const combine =
    findStage(
      effectiveSession,
      "combine"
    );


  const validate =
    findStage(
      effectiveSession,
      "validate"
    );


  const rootDatasetIds =
    snapshot
      .selected_analysis_dataset_ids;


  const analysisOutputDatasetIds =
    snapshot
      .analysis_output_dataset_ids;


  const validatedDatasetIds =
    snapshot
      .validated_analysis_dataset_ids;


  const outputSelected =
    analysisOutputDatasetIds.length >
    0;


  const validatedDatasetSet =
    new Set(
      validatedDatasetIds
    );


  const allOutputsValidated =
    outputSelected &&
    analysisOutputDatasetIds.every(
      (
        datasetId
      ) =>
        validatedDatasetSet.has(
          datasetId
        )
    );


  const ready =
    snapshot.ready_for_analysis;


  const candidates =
    candidatesResponse
      ?.candidates ??
    [];


  const materializedStageSet =
    new Set(
      candidates.map(
        (
          candidate
        ) =>
          candidate.stage
      )
    );


  const rootSourceRows =
    candidates.length >
      0
      ? candidates
          .filter(
            (
              candidate
            ) =>
              rootDatasetIds.includes(
                candidate.dataset_id
              )
          )
          .reduce(
            (
              total,
              candidate
            ) =>
              total +
              candidate.rows,
            0
          )
      : null;


  const analysisOutputRows =
    candidates.length >
      0 &&
    analysisOutputDatasetIds.length >
      0
      ? candidates
          .filter(
            (
              candidate
            ) =>
              analysisOutputDatasetIds.includes(
                candidate.dataset_id
              )
          )
          .reduce(
            (
              total,
              candidate
            ) =>
              total +
              candidate.rows,
            0
          )
      : null;


  const terminalDatasetIds =
    terminalCandidateDatasetIds(
      candidates
    );


  const terminalDatasetIdSet =
    new Set(
      terminalDatasetIds
    );


  const supersededDatasetIdSet =
    (() => {
      const terminalAncestors =
        new Set<
          string
        >();


      for (
        const terminalDatasetId
        of terminalDatasetIds
      ) {
        for (
          const ancestorId
          of collectAncestorDatasetIds(
            terminalDatasetId,
            candidates
          )
        ) {
          terminalAncestors.add(
            ancestorId
          );
        }
      }


      return terminalAncestors;
    })();


  const draftChanged =
    !sameDatasetSelection(
      draftDatasetIds,
      analysisOutputDatasetIds
    );


  /*
   * Candidate metadata must correspond to the same
   * server-owned Preparation revision that is currently
   * displayed.
   *
   * This prevents VALIDATE from becoming available for a
   * stale candidate snapshot during CLEAN / TRANSFORM /
   * COMBINE transitions.
   */
  const candidateScopeCurrent =
    candidatesResponse !==
      null &&
    candidatesResponse.revision ===
      effectiveSession.revision;


  /*
   * The browser may have a draft selection that differs from
   * the committed analytical scope.
   *
   * VALIDATE must never run against the old committed scope
   * while the UI is showing a newer uncommitted choice.
   */
  const selectionSynchronized =
    candidateScopeCurrent &&
    !candidateLoading &&
    candidateError ===
      null &&
    !draftChanged;


  const selectionLocked =
    ready ||
    candidatesResponse
      ?.locked ===
      true;


  const canValidate =
    !ready &&
    outputSelected &&
    selectionSynchronized &&
    snapshot.next_stage ===
      "validate";


  const canCommitSelection =
    !selectionLocked &&
    !selectionLoading &&
    draftDatasetIds.length >
      0 &&
    draftChanged;


  const filenameById =
    new Map(
      candidates.map(
        (
          candidate
        ) => [
          candidate.dataset_id,
          candidate.dataset_filename,
        ]
      )
    );


  const selectedCandidateNames =
    analysisOutputDatasetIds.map(
      (
        datasetId
      ) =>
        filenameById.get(
          datasetId
        ) ??
        datasetId
    );


  function toggleCandidate(
    datasetId:
      string
  ) {
    if (
      selectionLocked ||
      selectionLoading
    ) {
      return;
    }


    setSelectionError(
      null
    );


    setDraftDatasetIds(
      (
        current
      ) => {
        if (
          current.includes(
            datasetId
          )
        ) {
          return current.filter(
            (
              currentId
            ) =>
              currentId !==
              datasetId
          );
        }


        const relatedDatasetIds =
          lineageRelatedDatasetIds(
            datasetId,
            candidates
          );


        return [
          ...current.filter(
            (
              currentId
            ) =>
              !relatedDatasetIds.has(
                currentId
              )
          ),

          datasetId,
        ];
      }
    );
  }


  async function handleCommitSelection() {
    if (
      !canCommitSelection
    ) {
      return;
    }


    setSelectionLoading(
      true
    );

    setSelectionError(
      null
    );


    try {
      const updatedSession =
        await selectPreparationAnalysisOutput(
          effectiveSession.workflow_id,
          draftDatasetIds
        );


      setLocalSession(
        updatedSession
      );


      setDraftDatasetIds(
        updatedSession
          .analysis_output_dataset_ids
      );


      const refreshedCandidates =
        await getPreparationAnalysisOutputCandidates(
          updatedSession.workflow_id
        );


      setCandidatesResponse(
        refreshedCandidates
      );
    } catch (
      caughtError
    ) {
      setSelectionError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "La sélection de la sortie analytique a échoué."
      );
    } finally {
      setSelectionLoading(
        false
      );
    }
  }


  const validationStatusText =
    ready
      ? "✓ PRÊT POUR ANALYSE"
      : validate?.status ===
          "blocked"
        ? "× VALIDATION BLOQUÉE"
        : draftChanged
          ? "SÉLECTION À ENREGISTRER"
          : !outputSelected
            ? "SORTIE FINALE À SÉLECTIONNER"
            : candidateLoading ||
                !candidateScopeCurrent
              ? "VÉRIFICATION DU SCOPE"
              : "VALIDATION EN ATTENTE";


  const headline =
    ready
      ? "Préparation validée pour l’analyse"
      : draftChanged
        ? "Enregistrez la sortie analytique finale"
        : !outputSelected
          ? "Choisissez la sortie analytique finale"
          : canValidate
            ? "La préparation peut être validée"
            : "Terminez les étapes précédentes";


  return (
    <section
      className={
        `${styles.panel} ${
          ready
            ? styles.panelReady
            : ""
        }`
      }
      aria-labelledby="preparation-finalization-title"
    >
      <header
        className={
          styles.header
        }
      >
        <div
          className={
            styles.headerCopy
          }
        >
          <span
            className={
              styles.eyebrow
            }
          >
            Validation finale
          </span>

          <h3
            id="preparation-finalization-title"
          >
            {
              headline
            }
          </h3>

          <p>
            L’analyste choisit la ou les sorties préparées
            qui entreront dans l’analyse. DataLens vérifie ensuite
            leur traçabilité, verrouille la sélection après validation
            et n’ouvre l’analyse qu’une fois les contrôles terminés.
          </p>
        </div>

        <span
          className={
            `${styles.readinessBadge} ${
              ready
                ? styles.readinessBadgeReady
                : validate?.status ===
                    "blocked"
                  ? styles.readinessBadgeBlocked
                  : ""
            }`
          }
        >
          {
            validationStatusText
          }
        </span>
      </header>


      <section
        className={
          styles.selectionSection
        }
        aria-labelledby="analysis-output-selection-title"
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
              Périmètre d’analyse
            </span>

            <h4
              id="analysis-output-selection-title"
            >
              Sorties disponibles
            </h4>

            <p>
              DataLens présélectionne les sorties finales de la
              chaîne de préparation. Une sortie issue du nettoyage,
              d’une transformation ou d’un assemblage remplace ses
              versions intermédiaires afin d’éviter d’analyser
              plusieurs fois la même information.
            </p>
          </div>

          <div
            className={
              styles.selectionCount
            }
          >
            <strong>
              {
                draftDatasetIds.length
              }
            </strong>

            <span>
              sélectionné
              {
                draftDatasetIds.length >
                1
                  ? "s"
                  : ""
              }
            </span>
          </div>
        </div>


        {
          candidateLoading &&
          candidates.length ===
            0
            ? (
                <div
                  className={
                    styles.loadingState
                  }
                >
                  Lecture des artefacts matérialisés…
                </div>
              )
            : null
        }


        {
          !candidateLoading &&
          candidates.length ===
            0 &&
          !candidateError
            ? (
                <div
                  className={
                    styles.emptyState
                  }
                >
                  Aucune sortie matérialisée n’est encore disponible.
                  Revenez aux étapes de préparation précédentes.
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
                    styles.candidateGrid
                  }
                >
                  {
                    candidates.map(
                      (
                        candidate
                      ) => {
                        const checked =
                          draftDatasetIds.includes(
                            candidate.dataset_id
                          );

                        const recommended =
                          terminalDatasetIdSet.has(
                            candidate.dataset_id
                          );

                        const superseded =
                          supersededDatasetIdSet.has(
                            candidate.dataset_id
                          );

                        const outputExplanation =
                          outputExplanations[
                            candidate.dataset_id
                          ];

                        const outputExplanationLoading =
                          outputExplanationLoadingIds.includes(
                            candidate.dataset_id
                          );


                        return (
                          <label
                            key={
                              candidate.dataset_id
                            }
                            className={
                              `${styles.candidate} ${
                                checked
                                  ? styles.candidateSelected
                                  : ""
                              } ${
                                candidate.is_validated
                                  ? styles.candidateValidated
                                  : ""
                              }`
                            }
                          >
                            <input
                              type="checkbox"
                              checked={
                                checked
                              }
                              disabled={
                                selectionLocked ||
                                selectionLoading
                              }
                              onChange={
                                () =>
                                  toggleCandidate(
                                    candidate.dataset_id
                                  )
                              }
                            />

                            <div
                              className={
                                styles.candidateMain
                              }
                            >
                              <div
                                className={
                                  styles.candidateTop
                                }
                              >
                                <div>
                                  <span
                                    className={
                                      styles.candidateStage
                                    }
                                  >
                                    {
                                      artifactStageLabel(
                                        candidate
                                      )
                                    }
                                  </span>

                                  <strong>
                                    {
                                      candidate.dataset_filename
                                    }
                                  </strong>
                                </div>

                                <div
                                  className={
                                    styles.candidateBadges
                                  }
                                >
                                  {
                                    recommended
                                      ? (
                                          <span>
                                            Sortie finale
                                          </span>
                                        )
                                      : superseded
                                        ? (
                                            <span>
                                              Intermédiaire
                                            </span>
                                          )
                                        : null
                                  }

                                  {
                                    candidate.is_root_dataset
                                      ? (
                                          <span>
                                            Source
                                          </span>
                                        )
                                      : null
                                  }

                                  {
                                    candidate.is_validated
                                      ? (
                                          <span
                                            className={
                                              styles.validatedBadge
                                            }
                                          >
                                            Certifié
                                          </span>
                                        )
                                      : null
                                  }
                                </div>
                              </div>

                              <div
                                className={
                                  styles.candidateMetrics
                                }
                              >
                                <span>
                                  {
                                    candidate.rows
                                      .toLocaleString(
                                        "fr-FR"
                                      )
                                  }
                                  {" lignes"}
                                </span>

                                <span>
                                  {
                                    candidate.columns
                                      .toLocaleString(
                                        "fr-FR"
                                      )
                                  }
                                  {" colonnes"}
                                </span>

                                <span>
                                  {
                                    candidate.dataset_id
                                  }
                                </span>
                              </div>

                              {
                                candidate
                                  .parent_dataset_ids
                                  .length >
                                0
                                  ? (
                                      <p
                                        className={
                                          styles.lineage
                                        }
                                      >
                                        Parents ·
                                        {" "}
                                        {
                                          candidate
                                            .parent_dataset_ids
                                            .join(
                                              " · "
                                            )
                                        }
                                      </p>
                                    )
                                  : (
                                      <p
                                        className={
                                          styles.lineage
                                        }
                                      >
                                        Dataset racine importé
                                      </p>
                                    )
                              }


                              {
                                recommended
                                  ? (
                                      <div
                                        className={
                                          styles.outputExplanation
                                        }
                                      >
                                        <div
                                          className={
                                            styles.outputExplanationHead
                                          }
                                        >
                                          <span>
                                            Pourquoi cette sortie ?
                                          </span>

                                          <strong>
                                            {
                                              outputExplanation
                                                ?.explanation
                                                ?.python_validated
                                                ? "Explication contrôlée"
                                                : outputExplanationLoading
                                                  ? "Le modèle local prépare l’explication…"
                                                  : "Preuves déterministes"
                                            }
                                          </strong>
                                        </div>


                                        {
                                          outputExplanation
                                            ?.explanation
                                            ? (
                                                <>
                                                  <h5>
                                                    {
                                                      outputExplanation
                                                        .explanation
                                                        .title
                                                    }
                                                  </h5>

                                                  <p>
                                                    {
                                                      outputExplanation
                                                        .explanation
                                                        .explanation
                                                    }
                                                  </p>

                                                  <p
                                                    className={
                                                      styles.outputExplanationMessage
                                                    }
                                                  >
                                                    {
                                                      outputExplanation
                                                        .explanation
                                                        .user_message
                                                    }
                                                  </p>


                                                  {
                                                    outputExplanation
                                                      .explanation
                                                      .cautions
                                                      .length >
                                                    0
                                                      ? (
                                                          <div
                                                            className={
                                                              styles.outputExplanationCautions
                                                            }
                                                          >
                                                            {
                                                              outputExplanation
                                                                .explanation
                                                                .cautions
                                                                .map(
                                                                  (
                                                                    caution
                                                                  ) => (
                                                                    <span
                                                                      key={
                                                                        caution
                                                                      }
                                                                    >
                                                                      {
                                                                        caution
                                                                      }
                                                                    </span>
                                                                  )
                                                                )
                                                            }
                                                          </div>
                                                        )
                                                      : null
                                                  }
                                                </>
                                              )
                                            : outputExplanation
                                                ?.ai_error
                                              ? (
                                                  <>
                                                    <p>
                                                      {
                                                        outputExplanation
                                                          .facts
                                                          .deterministic_reasons[
                                                            0
                                                          ] ??
                                                        "Cette sortie est terminale dans la lineage courante."
                                                      }
                                                    </p>

                                                    <small
                                                      className={
                                                        styles.outputExplanationFallback
                                                      }
                                                    >
                                                      Explication du modèle local indisponible. La recommandation déterministe reste valide.
                                                    </small>
                                                  </>
                                                )
                                              : outputExplanationLoading
                                                ? (
                                                    <p>
                                                      La recommandation est déjà établie par le moteur déterministe.
                                                      Le modèle local transforme maintenant ces preuves
                                                      en une explication courte pour l’analyste.
                                                    </p>
                                                  )
                                                : (
                                                    <p>
                                                      Cette sortie est la version finale de la chaîne de
                                                      préparation et remplace ses versions intermédiaires
                                                      dans le périmètre analytique.
                                                    </p>
                                                  )
                                        }
                                      </div>
                                    )
                                  : null
                              }
                            </div>

                            <span
                              className={
                                styles.checkVisual
                              }
                              aria-hidden="true"
                            >
                              {
                                checked
                                  ? "✓"
                                  : ""
                              }
                            </span>
                          </label>
                        );
                      }
                    )
                  }
                </div>
              )
            : null
        }


        {
          candidateError
            ? (
                <p
                  className={
                    styles.error
                  }
                >
                  {
                    candidateError
                  }
                </p>
              )
            : null
        }


        {
          selectionError
            ? (
                <p
                  className={
                    styles.error
                  }
                >
                  {
                    selectionError
                  }
                </p>
              )
            : null
        }


        <div
          className={
            styles.selectionFooter
          }
        >
          <div>
            {
              selectionLocked
                ? (
                    <p>
                      La sélection est verrouillée après validation.
                    </p>
                  )
                : outputSelected
                  ? (
                      <p>
                        Périmètre actuellement enregistré ·
                        {" "}
                        {
                          selectedCandidateNames.join(
                            " · "
                          )
                        }
                      </p>
                    )
                  : (
                      <p>
                        {
                          draftDatasetIds.length >
                          0
                            ? (
                                "DataLens a présélectionné la frontière " +
                                "finale de la lineage. Confirmez-la pour " +
                                "l’enregistrer côté serveur."
                              )
                            : (
                                "Aucune sortie finale n’est encore committée."
                              )
                        }
                      </p>
                    )
            }

            {
              draftChanged
                ? (
                    <small>
                      Les changements ci-dessus ne sont pas encore enregistrés dans le workflow.
                    </small>
                  )
                : null
            }
          </div>

          <button
            type="button"
            className={
              styles.secondaryButton
            }
            disabled={
              !canCommitSelection
            }
            onClick={
              handleCommitSelection
            }
          >
            {
              selectionLoading
                ? "Enregistrement…"
                : outputSelected
                  ? "Mettre à jour la sélection"
                  : "Confirmer la sélection"
            }
          </button>
        </div>
      </section>


      <div
        className={
          styles.statusGrid
        }
      >
        <article>
          <span>
            Sources importées
          </span>

          <strong>
            {
              rootDatasetIds.length
            }
          </strong>
        </article>

        <article>
          <span>
            Lignes sources inspectées
          </span>

          <strong>
            {
              rootSourceRows !==
                null
                ? rootSourceRows
                    .toLocaleString(
                      "fr-FR"
                    )
                : "—"
            }
          </strong>
        </article>

        <article>
          <span>
            Lignes de sortie analytique
          </span>

          <strong>
            {
              analysisOutputRows !==
                null
                ? analysisOutputRows
                    .toLocaleString(
                      "fr-FR"
                    )
                : "—"
            }
          </strong>
        </article>

        <article>
          <span>
            Nettoyage
          </span>

          <strong>
            {
              stageStatusLabel(
                clean?.status,
                "clean",
                materializedStageSet.has(
                  "clean"
                )
              )
            }
          </strong>
        </article>

        <article>
          <span>
            Transformation
          </span>

          <strong>
            {
              stageStatusLabel(
                transform?.status,
                "transform",
                materializedStageSet.has(
                  "transform"
                )
              )
            }
          </strong>
        </article>

        <article>
          <span>
            Combinaison
          </span>

          <strong>
            {
              stageStatusLabel(
                combine?.status,
                "combine",
                materializedStageSet.has(
                  "combine"
                )
              )
            }
          </strong>
        </article>

        <article>
          <span>
            Sorties certifiées
          </span>

          <strong>
            {
              validatedDatasetIds.length
            }
            {" / "}
            {
              analysisOutputDatasetIds.length
            }
          </strong>
        </article>

        <article>
          <span>
            Validation
          </span>

          <strong>
            {
              stageStatusLabel(
                validate?.status,
                "validate"
              )
            }
          </strong>
        </article>
      </div>


      {
        ready &&
        !allOutputsValidated
          ? (
              <p
                className={
                  styles.error
                }
              >
                Incohérence détectée : le workflow se déclare
                prêt alors que toutes les sorties analytiques
                finales ne figurent pas dans le scope certifié.
              </p>
            )
          : null
      }


      {
        error
          ? (
              <p
                className={
                  styles.error
                }
              >
                {
                  error
                }
              </p>
            )
          : null
      }


      <footer
        className={
          styles.validationFooter
        }
      >
        <div>
          <strong>
            {
              ready
                ? "Analyse déverrouillée"
                : draftChanged
                  ? "Sélection non enregistrée"
                  : canValidate
                    ? "Scope final enregistré"
                    : outputSelected
                      ? "Préparation encore incomplète"
                      : "Sélection finale requise"
            }
          </strong>

          <p>
            {
              ready
                ? (
                    "Le readiness gate autorise maintenant l’analyse " +
                    "des sorties certifiées."
                  )
                : draftChanged
                  ? (
                      "Enregistrez la sélection ci-dessus avant de lancer " +
                      "la validation finale."
                    )
                  : candidateLoading ||
                      !candidateScopeCurrent
                    ? (
                        "DataLens vérifie que le scope analytique " +
                        "correspond à la révision serveur courante."
                      )
                    : candidateError
                      ? (
                          "Le scope analytique n’a pas pu être vérifié. " +
                          "La validation reste verrouillée."
                        )
                      : canValidate
                        ? (
                            "Le serveur peut maintenant contrôler la lineage, " +
                            "les preuves et l’existence des artefacts."
                          )
                        : !outputSelected
                          ? (
                              "Choisissez puis confirmez au moins une sortie " +
                              "avant d’exécuter VALIDATE."
                            )
                          : snapshot.next_stage
                            ? (
                                `Prochaine étape serveur : ${snapshot.next_stage}.`
                              )
                            : (
                                "La préparation n’est pas encore validable."
                              )
            }
          </p>
        </div>

        <button
          type="button"
          className={
            styles.primaryButton
          }
          onClick={
            onValidate
          }
          disabled={
            ready ||
            loading ||
            selectionLoading ||
            !canValidate
          }
        >
          {
            ready
              ? "Préparation validée"
              : loading
                ? "Validation…"
                : draftChanged
                  ? "Enregistrez d’abord la sélection"
                  : candidateLoading ||
                      !candidateScopeCurrent
                    ? "Vérification du scope…"
                    : "Valider la préparation"
          }
        </button>
      </footer>
    </section>
  );
}