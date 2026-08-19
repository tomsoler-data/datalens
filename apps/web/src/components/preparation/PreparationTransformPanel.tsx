"use client";

import {
  useEffect,
  useRef,
  useState,
} from "react";

import {
  approvePreparationCombine,
  continuePreparationWithoutSurrogate,
  createPreparationSurrogateKey,
  discoverPreparationCombine,
  getPreparationAnalysisOutputCandidates,
  inspectPreparationIdentity,
} from "./preparationApi";

import type {
  PreparationAnalysisOutputCandidate,
  PreparationCombineDiscoveryView,
  PreparationCombineExecutionResponse,
  PreparationCombineIntent,
  PreparationIdentityInspectResponse,
  PreparationSessionView,
  PreparationStageRecord,
} from "./preparationTypes";

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


type IdentityInspectionItem = {
  response:
    PreparationIdentityInspectResponse;

  aiLoading:
    boolean;
};


function findStage(
  session:
    PreparationSessionView |
    null,

  stageName:
    PreparationStageRecord[
      "stage"
    ]
): PreparationStageRecord | null {
  if (
    session ===
    null
  ) {
    return null;
  }


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


function hasCombineDiscoveryEvidence(
  stage:
    PreparationStageRecord |
    null
): boolean {
  return (
    stage
      ?.evidence_refs
      .some(
        (
          reference
        ) =>
          reference.startsWith(
            "combine_service:"
          )
      ) ??
    false
  );
}


function terminalCandidateDatasetIds(
  candidates:
    PreparationAnalysisOutputCandidate[]
): string[] {
  /*
   * Return the terminal materialized frontier.
   *
   * Important:
   * CLEAN / TRANSFORM can materialize an artifact in-place:
   *
   *   dataset_id = "orders"
   *   parent_dataset_ids = ["orders"]
   *
   * That self-parent is lineage evidence, not proof that the
   * current artifact has been superseded. It must therefore be
   * ignored, exactly like the backend COMBINE frontier does.
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


function identityStatusLabel(
  response:
    PreparationIdentityInspectResponse
): string {
  switch (
    response.report.status
  ) {
    case "single_key":
      return "Clé simple détectée";

    case "composite_key":
      return "Clé composite détectée";

    default:
      return "Clé technique suggérée";
  }
}


function identityCandidateLabel(
  response:
    PreparationIdentityInspectResponse
): string {
  const candidate =
    response
      .report
      .preferred_candidate;


  if (
    candidate !==
    null
  ) {
    return candidate.columns.join(
      " + "
    );
  }


  return (
    response
      .report
      .suggested_surrogate_column ??
    "Aucune"
  );
}


function statusLabel(
  stage:
    PreparationStageRecord |
    null
): string {
  switch (
    stage?.status
  ) {
    case "passed":
      return "Validé";

    case "skipped":
      return "Non requis";

    case "review_required":
      return "Revue requise";

    case "blocked":
      return "Bloqué";

    default:
      return "À faire";
  }
}


function statusClass(
  stage:
    PreparationStageRecord |
    null
): string {
  switch (
    stage?.status
  ) {
    case "passed":
      return styles.success;

    case "review_required":
    case "blocked":
      return styles.attention;

    case "skipped":
      return styles.skipped;

    default:
      return styles.pending;
  }
}


function joinTypeLabel(
  value:
    string
): string {
  switch (
    value.toLowerCase()
  ) {
    case "left":
      return "LEFT JOIN";

    case "inner":
      return "INNER JOIN";

    case "right":
      return "RIGHT JOIN";

    case "outer":
      return "FULL OUTER JOIN";

    default:
      return value;
  }
}


function cardinalityLabel(
  value:
    string
): string {
  switch (
    value.toLowerCase()
  ) {
    case "many_to_one":
      return "Plusieurs vers un";

    case "one_to_one":
      return "Un vers un";

    case "one_to_many":
      return "Un vers plusieurs";

    case "many_to_many":
      return "Plusieurs vers plusieurs";

    default:
      return value;
  }
}


function joinKeyLabel(
  intent:
    PreparationCombineIntent
): string {
  if (
    intent.keys.length ===
    0
  ) {
    return "Clé non disponible";
  }


  return intent.keys
    .map(
      (
        key
      ) =>
        key.left_column ===
          key.right_column
          ? key.left_column
          : `${
              key.left_column
            } = ${
              key.right_column
            }`
    )
    .join(
      " + "
    );
}


function recordString(
  value:
    unknown
): string | null {
  return (
    typeof value ===
      "string" &&
    value.trim()
  )
    ? value
    : null;
}


function recordStringArray(
  value:
    unknown
): string[] {
  if (
    !Array.isArray(
      value
    )
  ) {
    return [];
  }


  return value.filter(
    (
      item
    ): item is string =>
      typeof item ===
      "string" &&
      Boolean(
        item.trim()
      )
  );
}


function planWarnings(
  discovery:
    PreparationCombineDiscoveryView |
    null
): string[] {
  const joins =
    discovery
      ?.plan
      ?.joins;


  if (
    !Array.isArray(
      joins
    ) ||
    joins.length ===
      0
  ) {
    return [];
  }


  const first =
    joins[
      0
    ];


  if (
    !first ||
    typeof first !==
      "object" ||
    Array.isArray(
      first
    )
  ) {
    return [];
  }


  const record =
    first as
      Record<
        string,
        unknown
      >;


  const rationale =
    recordString(
      record.rationale
    );


  return [
    ...(
      rationale
        ? [
            rationale,
          ]
        : []
    ),

    ...recordStringArray(
      record.warnings
    ),
  ];
}


function validationPassed(
  execution:
    PreparationCombineExecutionResponse |
    null
): boolean {
  if (
    execution ===
    null
  ) {
    return false;
  }


  return (
    execution
      .validation[
        "valid_for_downstream"
      ] ===
    true
  );
}


export default function PreparationTransformPanel({
  session,
  onSessionChange,
}: PreparationTransformPanelProps) {
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
    discovery,
    setDiscovery,
  ] = useState<
    PreparationCombineDiscoveryView |
    null
  >(
    null
  );


  const [
    lastExecution,
    setLastExecution,
  ] = useState<
    PreparationCombineExecutionResponse |
    null
  >(
    null
  );


  const [
    discovering,
    setDiscovering,
  ] = useState(
    false
  );


  const [
    approving,
    setApproving,
  ] = useState(
    false
  );


  const [
    error,
    setError,
  ] = useState<
    string |
    null
  >(
    null
  );


  const [
    identityItems,
    setIdentityItems,
  ] = useState<
    Record<
      string,
      IdentityInspectionItem
    >
  >(
    {}
  );


  const [
    identityLoading,
    setIdentityLoading,
  ] = useState(
    false
  );


  const [
    identityError,
    setIdentityError,
  ] = useState<
    string |
    null
  >(
    null
  );


  const [
    creatingSurrogateDatasetId,
    setCreatingSurrogateDatasetId,
  ] = useState<
    string |
    null
  >(
    null
  );


  const [
    continuingIdentityDatasetId,
    setContinuingIdentityDatasetId,
  ] = useState<
    string |
    null
  >(
    null
  );


  const identityInspectionKeyRef =
    useRef<
      string |
      null
    >(
      null
    );


  const automaticDiscoveryKeyRef =
    useRef<
      string |
      null
    >(
      null
    );


  useEffect(
    () => {
      setLocalSession(
        session
      );


      setDiscovery(
        null
      );


      setLastExecution(
        null
      );


      setError(
        null
      );


      setIdentityItems(
        {}
      );


      setIdentityLoading(
        false
      );


      setIdentityError(
        null
      );


      setCreatingSurrogateDatasetId(
        null
      );


      setContinuingIdentityDatasetId(
        null
      );


      identityInspectionKeyRef
        .current =
          null;


      automaticDiscoveryKeyRef
        .current =
          null;
    },
    [
      session
        ?.workflow_id,
    ]
  );


  useEffect(
    () => {
      if (
        session !==
        null
      ) {
        setLocalSession(
          session
        );
      }
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
        return;
      }


      const currentTransform =
        findStage(
          currentSession,
          "transform"
        );

      const currentCombine =
        findStage(
          currentSession,
          "combine"
        );

      const currentValidate =
        findStage(
          currentSession,
          "validate"
        );


      const transformIsResolved =
        currentTransform?.status ===
          "passed" ||
        currentTransform?.status ===
          "skipped";


      const validationLocksPreparation =
        currentValidate?.status ===
          "passed" ||
        currentSession
          .snapshot
          .ready_for_analysis;


      const combineAlreadyStarted =
        hasCombineDiscoveryEvidence(
          currentCombine
        );


      if (
        !transformIsResolved ||
        validationLocksPreparation ||
        combineAlreadyStarted
      ) {
        return;
      }


      const orchestrationKey =
        `${
          currentSession.workflow_id
        }:${
          currentSession.revision
        }`;


      if (
        identityInspectionKeyRef
          .current ===
        orchestrationKey
      ) {
        return;
      }


      identityInspectionKeyRef
        .current =
          orchestrationKey;


      setIdentityLoading(
        true
      );

      setIdentityError(
        null
      );


      void (
        async () => {
          try {
            const candidatesResponse =
              await getPreparationAnalysisOutputCandidates(
                currentSession
                  .workflow_id
              );


            const terminalIds =
              new Set(
                terminalCandidateDatasetIds(
                  candidatesResponse
                    .candidates
                )
              );


            const targets =
              candidatesResponse
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


            if (
              candidatesResponse
                .candidates
                .length >
                0 &&
              targets.length ===
                0 &&
              !candidatesResponse
                .candidates
                .every(
                  (
                    candidate
                  ) =>
                    candidate.stage ===
                    "combine"
                )
            ) {
              throw (
                new Error(
                  "DataLens n’a trouvé aucun artefact terminal à inspecter pour l’identité des lignes."
                )
              );
            }


            const deterministicResponses =
              await Promise.all(
                targets.map(
                  (
                    candidate
                  ) =>
                    inspectPreparationIdentity(
                      currentSession
                        .workflow_id,

                      candidate
                        .dataset_id,

                      false
                    )
                )
              );


            setIdentityItems(
              Object.fromEntries(
                deterministicResponses.map(
                  (
                    response
                  ) => [
                    response.dataset_id,
                    {
                      response,
                      aiLoading:
                        true,
                    },
                  ]
                )
              )
            );


            setIdentityLoading(
              false
            );


            const unresolvedRecommendations =
              deterministicResponses.filter(
                (
                  response
                ) =>
                  !response
                    .identity_resolved
              );


            if (
              deterministicResponses.length >
                0 &&
              unresolvedRecommendations.length ===
                0
            ) {
              void startAutomaticCombine(
                currentSession
              );
            }


            for (
              const deterministicResponse
              of deterministicResponses
            ) {
              void inspectPreparationIdentity(
                currentSession
                  .workflow_id,

                deterministicResponse
                  .dataset_id,

                true
              )
                .then(
                  (
                    response
                  ) => {
                    setIdentityItems(
                      (
                        current
                      ) => ({
                        ...current,

                        [
                          response.dataset_id
                        ]: {
                          response,
                          aiLoading:
                            false,
                        },
                      })
                    );
                  }
                )
                .catch(
                  (
                    caughtError
                  ) => {
                    setIdentityItems(
                      (
                        current
                      ) => {
                        const existing =
                          current[
                            deterministicResponse
                              .dataset_id
                          ];


                        if (
                          !existing
                        ) {
                          return current;
                        }


                        return {
                          ...current,

                          [
                            deterministicResponse
                              .dataset_id
                          ]: {
                            ...existing,
                            aiLoading:
                              false,

                            response: {
                              ...existing.response,

                              ai_error:
                                caughtError
                                  instanceof Error
                                  ? caughtError.message
                                  : "Explication du modèle local indisponible.",
                            },
                          },
                        };
                      }
                    );
                  }
                );
            }
          } catch (
            caughtError
          ) {
            identityInspectionKeyRef
              .current =
                null;

            setIdentityLoading(
              false
            );

            setIdentityError(
              caughtError
                instanceof Error
                ? caughtError.message
                : "L’inspection de l’identité des lignes a échoué."
            );
          }
        }
      )();
    },
    [
      session,
    ]
  );


  if (
    session ===
    null
  ) {
    return null;
  }


  const effectiveSession =
    localSession ??
    session;


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


  const transformResolved =
    transform?.status ===
      "passed" ||
    transform?.status ===
      "skipped";


  const combineResolved =
    combine?.status ===
      "passed" ||
    combine?.status ===
      "skipped";


  const multipleSourceDatasets =
    effectiveSession
      .selected_analysis_dataset_ids
      .length >
    1;


  const combineDiscoveryRecorded =
    hasCombineDiscoveryEvidence(
      combine
    );


  const combineDiscoveryPending =
    multipleSourceDatasets &&
    !combineDiscoveryRecorded;


  const combineLocked =
    validate?.status ===
      "passed" ||
    effectiveSession
      .snapshot
      .ready_for_analysis;


  const allSkipped =
    transform?.status ===
      "skipped" &&
    combine?.status ===
      "skipped" &&
    !combineDiscoveryPending;


  const currentIntent =
    discovery
      ?.intent ??
    null;


  const warnings =
    planWarnings(
      discovery
    );


  const identityInspectionItems =
    Object.values(
      identityItems
    );


  const unresolvedIdentityItems =
    identityInspectionItems.filter(
      (
        item
      ) =>
        !item.response
          .identity_resolved
    );


  const identityReadyForCombine =
    !identityLoading &&
    identityError ===
      null &&
    identityInspectionItems.length >
      0 &&
    unresolvedIdentityItems.length ===
      0;


  const canDiscover =
    transformResolved &&
    identityReadyForCombine &&
    !combineLocked &&
    !discovering &&
    !approving;


  const canApprove =
    !combineLocked &&
    !discovering &&
    !approving &&
    discovery
      ?.has_candidate ===
      true &&
    discovery
      .ready_for_approval ===
      true &&
    currentIntent !==
      null;


  function synchronizeSession(
    nextSession:
      PreparationSessionView
  ) {
    setLocalSession(
      nextSession
    );


    onSessionChange?.(
      nextSession
    );
  }


  async function startAutomaticCombine(
    currentSession:
      PreparationSessionView
  ) {
    const currentCombine =
      findStage(
        currentSession,
        "combine"
      );


    if (
      hasCombineDiscoveryEvidence(
        currentCombine
      ) ||
      currentSession
        .selected_analysis_dataset_ids
        .length <=
        1
    ) {
      return;
    }


    const discoveryKey =
      `${
        currentSession.workflow_id
      }:${
        currentSession.revision
      }`;


    if (
      automaticDiscoveryKeyRef
        .current ===
      discoveryKey
    ) {
      return;
    }


    automaticDiscoveryKeyRef
      .current =
        discoveryKey;


    setDiscovering(
      true
    );

    setError(
      null
    );

    setLastExecution(
      null
    );


    try {
      const response =
        await discoverPreparationCombine(
          currentSession
            .workflow_id
        );


      setDiscovery(
        response.discovery
      );


      synchronizeSession(
        response.session
      );
    } catch (
      caughtError
    ) {
      automaticDiscoveryKeyRef
        .current =
          null;

      setError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "La détection automatique des relations entre datasets a échoué."
      );
    } finally {
      setDiscovering(
        false
      );
    }
  }


  async function handleCreateSurrogate(
    response:
      PreparationIdentityInspectResponse
  ) {
    const requestId =
      response
        .surrogate_request_id;


    if (
      !requestId ||
      !response
        .can_create_surrogate
    ) {
      return;
    }


    setCreatingSurrogateDatasetId(
      response.dataset_id
    );

    setIdentityError(
      null
    );


    try {
      const creation =
        await createPreparationSurrogateKey(
          effectiveSession
            .workflow_id,

          response
            .dataset_id,

          requestId
        );


      setIdentityItems(
        {}
      );


      identityInspectionKeyRef
        .current =
          null;

      automaticDiscoveryKeyRef
        .current =
          null;


      synchronizeSession(
        creation.session
      );
    } catch (
      caughtError
    ) {
      setIdentityError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "La création de la clé technique a échoué."
      );
    } finally {
      setCreatingSurrogateDatasetId(
        null
      );
    }
  }


  async function handleContinueWithoutSurrogate(
    response:
      PreparationIdentityInspectResponse
  ) {
    const requestId =
      response
        .surrogate_request_id;


    if (
      !requestId ||
      !response
        .can_continue_without_surrogate
    ) {
      return;
    }


    setContinuingIdentityDatasetId(
      response.dataset_id
    );

    setIdentityError(
      null
    );


    try {
      await continuePreparationWithoutSurrogate(
        effectiveSession
          .workflow_id,

        response
          .dataset_id,

        requestId
      );


      const refreshed =
        await inspectPreparationIdentity(
          effectiveSession
            .workflow_id,

          response
            .dataset_id,

          true
        );


      setIdentityItems(
        (
          current
        ) => ({
          ...current,

          [
            refreshed.dataset_id
          ]: {
            response:
              refreshed,

            aiLoading:
              false,
          },
        })
      );


      const remainingUnresolved =
        unresolvedIdentityItems.filter(
          (
            item
          ) =>
            item.response
              .dataset_id !==
            response.dataset_id
        );


      if (
        remainingUnresolved.length ===
          0
      ) {
        void startAutomaticCombine(
          effectiveSession
        );
      }
    } catch (
      caughtError
    ) {
      setIdentityError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "La décision de continuer sans clé technique a échoué."
      );
    } finally {
      setContinuingIdentityDatasetId(
        null
      );
    }
  }


  async function handleDiscover() {
    if (
      !canDiscover
    ) {
      return;
    }


    setDiscovering(
      true
    );

    setError(
      null
    );

    setLastExecution(
      null
    );


    try {
      const response =
        await discoverPreparationCombine(
          effectiveSession
            .workflow_id
        );


      setDiscovery(
        response.discovery
      );


      synchronizeSession(
        response.session
      );
    } catch (
      caughtError
    ) {
      setError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "La détection des relations entre datasets a échoué."
      );
    } finally {
      setDiscovering(
        false
      );
    }
  }


  async function handleApprove() {
    /*
     * Approval intentionally does not depend on canDiscover.
     *
     * A COMBINE discovery changes the server-owned Preparation
     * revision. That revision change must not invalidate the
     * already-issued join proposal in the UI.
     *
     * Security remains server-owned:
     * approvePreparationCombine() rediscoveres the current
     * candidate, re-runs the Identity gate and verifies the
     * exact request_id before executing anything.
     */
    if (
      !canApprove ||
      currentIntent ===
        null
    ) {
      return;
    }


    setApproving(
      true
    );

    setError(
      null
    );


    try {
      const response =
        await approvePreparationCombine(
          effectiveSession
            .workflow_id,

          currentIntent
            .request_id,

          "Jointure approuvée depuis l’étape Preparation de DataLens."
        );


      setLastExecution(
        response
      );


      setDiscovery(
        response.next_discovery
      );


      synchronizeSession(
        response.session
      );
    } catch (
      caughtError
    ) {
      setError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "La jointure contrôlée a échoué."
      );
    } finally {
      setApproving(
        false
      );
    }
  }


  const headline =
    unresolvedIdentityItems.length >
      0
      ? "Résolvez d’abord l’identité des lignes"
      : identityLoading
        ? "Vérification de l’identité des lignes"
        : combineLocked
      ? "Combinaison verrouillée après validation"
      : discovery
          ?.has_candidate ===
          true
        ? discovery
            .ready_for_approval
          ? "Une relation sûre attend votre approbation"
          : "Une relation a été détectée mais reste bloquée"
        : discovery
            ?.has_candidate ===
            false
          ? combine?.status ===
              "passed"
            ? "Les combinaisons nécessaires sont terminées"
            : "Aucune combinaison supplémentaire n’est requise"
          : combineDiscoveryPending
            ? discovering
              ? "Recherche automatique des relations"
              : "Vérification des relations à effectuer"
            : "Rechercher les relations entre datasets";


  return (
    <section
      className={
        styles.panel
      }
      aria-labelledby="preparation-transform-title"
    >
      <div
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
            Transformer et combiner
          </h3>

          <p>
            DataLens vérifie d’abord l’identité des lignes, puis
            sépare les transformations des jointures. Python produit
            les preuves, le modèle local les explique et toute mutation
            reste sous contrôle de l’analyste.
          </p>
        </div>

        <span
          className={
            `${styles.badge} ${
              combine?.status ===
                "passed"
                ? styles.success
                : combine?.status ===
                    "review_required" ||
                  combine?.status ===
                    "blocked"
                  ? styles.attention
                  : allSkipped
                    ? styles.skipped
                    : ""
            }`
          }
        >
          {
            combineLocked
              ? "VERROUILLÉ"
              : combine?.status ===
                  "review_required"
                ? "APPROBATION REQUISE"
                : combine?.status ===
                    "passed"
                  ? "COMBINAISON VALIDÉE"
                  : allSkipped
                    ? "À VÉRIFIER"
                    : "CONTRÔLÉ"
          }
        </span>
      </div>


      <div
        className={
          styles.grid
        }
      >
        <article
          className={
            styles.card
          }
        >
          <div
            className={
              styles.cardTop
            }
          >
            <strong>
              Transformations
            </strong>

            <span
              className={
                `${styles.state} ${
                  statusClass(
                    transform
                  )
                }`
              }
            >
              {
                statusLabel(
                  transform
                )
              }
            </span>
          </div>

          <p>
            Typage, variables dérivées, agrégations et autres opérations
            structurelles restent exécutés par les moteurs Python
            déterministes.
          </p>

          {
            transform
              ?.blocking_reasons
              .length
              ? (
                  <div
                    className={
                      styles.reasons
                    }
                  >
                    {
                      transform
                        .blocking_reasons
                        .map(
                          (
                            reason
                          ) => (
                            <span
                              key={
                                reason
                              }
                            >
                              {
                                reason
                              }
                            </span>
                          )
                        )
                    }
                  </div>
                )
              : null
          }
        </article>


        <article
          className={
            styles.card
          }
        >
          <div
            className={
              styles.cardTop
            }
          >
            <strong>
              Combinaison
            </strong>

            <span
              className={
                `${styles.state} ${
                  statusClass(
                    combine
                  )
                }`
              }
            >
              {
                combineDiscoveryPending
                  ? discovering
                    ? "Vérification…"
                    : "À vérifier"
                  : statusLabel(
                      combine
                    )
              }
            </span>
          </div>

          <p>
            Les clés, le type de jointure et la cardinalité ne sont
            jamais envoyés par le navigateur. Ils sont dérivés et
            validés côté serveur.
          </p>

          {
            combine
              ?.blocking_reasons
              .length
              ? (
                  <div
                    className={
                      styles.reasons
                    }
                  >
                    {
                      combine
                        .blocking_reasons
                        .map(
                          (
                            reason
                          ) => (
                            <span
                              key={
                                reason
                              }
                            >
                              {
                                reason
                              }
                            </span>
                          )
                        )
                    }
                  </div>
                )
              : null
          }
        </article>
      </div>


      <div
        className={
          styles.identityWorkspace
        }
      >
        <div
          className={
            styles.identityWorkspaceHead
          }
        >
          <div>
            <span
              className={
                styles.eyebrow
              }
            >
              Identité des lignes
            </span>

            <strong>
              {
                identityLoading
                  ? "Analyse déterministe en cours"
                  : unresolvedIdentityItems.length >
                      0
                    ? `${
                        unresolvedIdentityItems.length
                      } recommandation${
                        unresolvedIdentityItems.length >
                          1
                          ? "s"
                          : ""
                      } à examiner`
                    : identityInspectionItems.length >
                        0
                      ? "Identité contrôlée"
                      : "Contrôle en attente"
              }
            </strong>

            <p>
              Python vérifie l’unicité et les clés candidates.
              Gemma reçoit uniquement ces faits structurés pour
              les expliquer. Toute décision ambiguë est enregistrée
              côté serveur avant que COMBINE puisse démarrer.
            </p>
          </div>

          <span
            className={
              `${styles.identityGate} ${
                unresolvedIdentityItems.length >
                  0
                  ? styles.attention
                  : identityInspectionItems.length >
                      0 &&
                    !identityLoading
                    ? styles.success
                    : styles.pending
              }`
            }
          >
            {
              identityLoading
                ? "VÉRIFICATION"
                : unresolvedIdentityItems.length >
                    0
                  ? "DÉCISION REQUISE"
                  : identityInspectionItems.length >
                      0
                    ? "IDENTITÉ RÉSOLUE"
                    : "EN ATTENTE"
            }
          </span>
        </div>


        {
          identityLoading &&
          identityInspectionItems.length ===
            0
            ? (
                <div
                  className={
                    styles.identityLoading
                  }
                >
                  <strong>
                    Inspection des artefacts actifs…
                  </strong>

                  <span>
                    DataLens calcule d’abord les preuves
                    d’unicité côté Python avant tout appel
                    au modèle local.
                  </span>
                </div>
              )
            : null
        }


        {
          identityInspectionItems.length >
            0
            ? (
                <div
                  className={
                    styles.identityGrid
                  }
                >
                  {
                    identityInspectionItems.map(
                      (
                        item
                      ) => {
                        const response =
                          item.response;

                        const continuedWithoutSurrogate =
                          response.resolution_kind ===
                          "continued_without_surrogate";

                        const creating =
                          creatingSurrogateDatasetId ===
                          response.dataset_id;

                        const continuing =
                          continuingIdentityDatasetId ===
                          response.dataset_id;

                        const preferred =
                          response
                            .report
                            .preferred_candidate;

                        const ai =
                          response.explanation;


                        return (
                          <article
                            key={
                              response.dataset_id
                            }
                            className={
                              `${styles.identityCard} ${
                                !response
                                  .identity_resolved
                                  ? styles.identityCardAttention
                                  : styles.identityCardResolved
                              }`
                            }
                          >
                            <div
                              className={
                                styles.identityCardHead
                              }
                            >
                              <div>
                                <span>
                                  {
                                    response.artifact_stage
                                      .toUpperCase()
                                  }
                                </span>

                                <strong
                                  title={
                                    response.dataset_id
                                  }
                                >
                                  {
                                    response
                                      .dataset_filename
                                  }
                                </strong>
                              </div>

                              <span
                                className={
                                  `${styles.identityStatus} ${
                                    !response
                                      .identity_resolved
                                      ? styles.attention
                                      : styles.success
                                  }`
                                }
                              >
                                {
                                  continuedWithoutSurrogate
                                    ? "CONTINUÉ SANS CLÉ"
                                    : response
                                        .identity_resolved
                                      ? identityStatusLabel(
                                          response
                                        )
                                      : "DÉCISION REQUISE"
                                }
                              </span>
                            </div>


                            <div
                              className={
                                styles.identityEvidence
                              }
                            >
                              <div>
                                <span>
                                  Clé / suggestion
                                </span>

                                <strong>
                                  {
                                    identityCandidateLabel(
                                      response
                                    )
                                  }
                                </strong>
                              </div>

                              <div>
                                <span>
                                  Unicité
                                </span>

                                <strong>
                                  {
                                    preferred
                                      ? `${
                                          preferred.unique_count
                                        } / ${
                                          preferred.row_count
                                        }`
                                      : "Aucune clé fiable"
                                  }
                                </strong>
                              </div>

                              <div>
                                <span>
                                  Valeurs manquantes
                                </span>

                                <strong>
                                  {
                                    preferred
                                      ? preferred
                                          .missing_row_count
                                      : "—"
                                  }
                                </strong>
                              </div>
                            </div>


                            <div
                              className={
                                styles.identityNarrative
                              }
                            >
                              <div
                                className={
                                  styles.identityNarrativeHead
                                }
                              >
                                <span>
                                  Modèle local
                                </span>

                                <strong>
                                  {
                                    item.aiLoading
                                      ? "Génération de l’explication…"
                                      : ai
                                        ? "Explication validée par Python"
                                        : "Preuves Python disponibles"
                                  }
                                </strong>
                              </div>

                              {
                                ai
                                  ? (
                                      <>
                                        <h4>
                                          {
                                            ai.title
                                          }
                                        </h4>

                                        <p>
                                          {
                                            ai.explanation
                                          }
                                        </p>

                                        <p
                                          className={
                                            styles.identityUserMessage
                                          }
                                        >
                                          {
                                            ai.user_message
                                          }
                                        </p>
                                      </>
                                    )
                                  : (
                                      <p>
                                        {
                                          response
                                            .report
                                            .reasons[
                                              0
                                            ] ??
                                          "Le contrôle déterministe est disponible."
                                        }
                                      </p>
                                    )
                              }

                              {
                                response.ai_error
                                  ? (
                                      <span
                                        className={
                                          styles.aiFallback
                                        }
                                      >
                                        Explication IA indisponible :
                                        {" "}
                                        {
                                          response.ai_error
                                        }
                                        {" "}
                                        La décision Python reste utilisable.
                                      </span>
                                    )
                                  : null
                              }

                              {
                                ai
                                  ?.cautions
                                  .length
                                  ? (
                                      <div
                                        className={
                                          styles.identityCautions
                                        }
                                      >
                                        {
                                          ai.cautions.map(
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
                            </div>


                            {
                              response
                                .can_create_surrogate &&
                              !response
                                .identity_resolved
                                ? (
                                    <div
                                      className={
                                        styles.identityActions
                                      }
                                    >
                                      <p>
                                        Une clé technique améliore la traçabilité
                                        des lignes mais ne devient jamais une clé
                                        de jointure entre datasets.
                                      </p>

                                      <div
                                        className={
                                          styles.identityActionButtons
                                        }
                                      >
                                        <button
                                          type="button"
                                          className={
                                            styles.secondaryButton
                                          }
                                          onClick={
                                            () =>
                                              handleContinueWithoutSurrogate(
                                                response
                                              )
                                          }
                                          disabled={
                                            creating ||
                                            continuing ||
                                            discovering ||
                                            approving
                                          }
                                        >
                                          {
                                            continuing
                                              ? "Enregistrement…"
                                              : "Continuer sans créer"
                                          }
                                        </button>

                                        <button
                                          type="button"
                                          className={
                                            styles.primaryButton
                                          }
                                          onClick={
                                            () =>
                                              handleCreateSurrogate(
                                                response
                                              )
                                          }
                                          disabled={
                                            creating ||
                                            continuing ||
                                            discovering ||
                                            approving
                                          }
                                        >
                                          {
                                            creating
                                              ? "Création…"
                                              : `Créer ${
                                                  response
                                                    .report
                                                    .suggested_surrogate_column ??
                                                  "la clé technique"
                                                }`
                                          }
                                        </button>
                                      </div>
                                    </div>
                                  )
                                : continuedWithoutSurrogate
                                  ? (
                                      <div
                                        className={
                                          styles.identityResolutionNotice
                                        }
                                      >
                                        <strong>
                                          Décision analyste enregistrée
                                        </strong>

                                        <span>
                                          DataLens peut poursuivre vers COMBINE
                                          sans créer de clé technique pour cet
                                          artefact. Cette décision est liée au
                                          rapport déterministe courant et deviendra
                                          caduque si le dataset change.
                                        </span>
                                      </div>
                                    )
                                  : null
                            }


                            {
                              response.mutation_locked
                                ? (
                                    <div
                                      className={
                                        styles.identityLock
                                      }
                                    >
                                      {
                                        response
                                          .mutation_lock_reason ??
                                        "Mutation verrouillée."
                                      }
                                    </div>
                                  )
                                : null
                            }
                          </article>
                        );
                      }
                    )
                  }
                </div>
              )
            : null
        }


        {
          identityError
            ? (
                <div
                  className={
                    styles.error
                  }
                >
                  <strong>
                    Inspection d’identité impossible
                  </strong>

                  <span>
                    {
                      identityError
                    }
                  </span>
                </div>
              )
            : null
        }


        <div
          className={
            styles.identityRule
          }
        >
          <strong>
            Règle de sécurité
          </strong>

          <span>
            Une clé technique identifie une ligne. Elle ne crée jamais
            artificiellement une relation entre deux fichiers. COMBINE
            reste verrouillé côté serveur tant que l’identité des artefacts
            actifs n’est pas résolue.
          </span>
        </div>
      </div>


      <div
        className={
          styles.combineWorkspace
        }
      >
        <div
          className={
            styles.combineWorkspaceHead
          }
        >
          <div>
            <span
              className={
                styles.eyebrow
              }
            >
              Jointures contrôlées
            </span>

            <strong>
              {
                headline
              }
            </strong>

            <p>
              {
                discovery
                  ?.reason ??
                (
                  combineDiscoveryPending
                    ? identityReadyForCombine
                      ? "L’identité des lignes est résolue. DataLens vérifie maintenant automatiquement si les datasets actifs partagent une relation déterministe sûre avant d’autoriser la Validation."
                      : "La recherche de relations commencera automatiquement après la résolution des recommandations d’identité de ligne."
                    : "Relancez la détection pour vérifier si les datasets actifs partagent encore une relation déterministe sûre."
                )
              }
            </p>
          </div>

          <button
            type="button"
            className={
              styles.secondaryButton
            }
            onClick={
              handleDiscover
            }
            disabled={
              !canDiscover
            }
          >
            {
              discovering
                ? "Détection automatique…"
                : !identityReadyForCombine
                  ? "Identité à résoudre"
                  : discovery ===
                      null &&
                    combineDiscoveryPending
                    ? "Vérification automatique"
                  : discovery ===
                      null
                    ? "Rechercher les relations"
                    : "Actualiser la proposition"
            }
          </button>
        </div>


        {
          lastExecution
            ? (
                <div
                  className={
                    `${styles.executionNotice} ${
                      validationPassed(
                        lastExecution
                      )
                        ? styles.executionNoticeSuccess
                        : ""
                    }`
                  }
                >
                  <div>
                    <span>
                      Dernière jointure matérialisée
                    </span>

                    <strong>
                      {
                        lastExecution
                          .output_dataset_filename
                      }
                    </strong>
                  </div>

                  <div
                    className={
                      styles.executionMetrics
                    }
                  >
                    <span>
                      {
                        lastExecution.rows
                      }
                      {" lignes"}
                    </span>

                    <span>
                      {
                        lastExecution.columns
                      }
                      {" colonnes"}
                    </span>

                    <span>
                      {
                        validationPassed(
                          lastExecution
                        )
                          ? "Validation post-jointure OK"
                          : "Validation à contrôler"
                      }
                    </span>
                  </div>
                </div>
              )
            : null
        }


        {
          currentIntent
            ? (
                <article
                  className={
                    `${styles.joinProposal} ${
                      discovery
                        ?.ready_for_approval
                        ? styles.joinProposalReady
                        : styles.joinProposalBlocked
                    }`
                  }
                >
                  <div
                    className={
                      styles.joinProposalHead
                    }
                  >
                    <div>
                      <span
                        className={
                          styles.eyebrow
                        }
                      >
                        Proposition serveur
                      </span>

                      <strong>
                        {
                          currentIntent
                            .left_dataset_filename
                        }
                        {" + "}
                        {
                          currentIntent
                            .right_dataset_filename
                        }
                      </strong>
                    </div>

                    <span
                      className={
                        discovery
                          ?.ready_for_approval
                          ? styles.proposalReady
                          : styles.proposalBlocked
                      }
                    >
                      {
                        discovery
                          ?.ready_for_approval
                          ? "PRÊTE À APPROUVER"
                          : "BLOQUÉE"
                      }
                    </span>
                  </div>


                  <div
                    className={
                      styles.joinFlow
                    }
                  >
                    <div
                      className={
                        styles.datasetNode
                      }
                    >
                      <span>
                        Dataset gauche
                      </span>

                      <strong
                        title={
                          currentIntent
                            .left_dataset_id
                        }
                      >
                        {
                          currentIntent
                            .left_dataset_filename
                        }
                      </strong>

                      <code>
                        {
                          currentIntent
                            .keys[
                              0
                            ]
                            ?.left_column ??
                          "—"
                        }
                      </code>
                    </div>


                    <div
                      className={
                        styles.joinOperator
                      }
                      aria-hidden="true"
                    >
                      <span>
                        {
                          joinTypeLabel(
                            currentIntent
                              .join_type
                          )
                        }
                      </span>

                      <strong>
                        =
                      </strong>
                    </div>


                    <div
                      className={
                        styles.datasetNode
                      }
                    >
                      <span>
                        Dataset droit
                      </span>

                      <strong
                        title={
                          currentIntent
                            .right_dataset_id
                        }
                      >
                        {
                          currentIntent
                            .right_dataset_filename
                        }
                      </strong>

                      <code>
                        {
                          currentIntent
                            .keys[
                              0
                            ]
                            ?.right_column ??
                          "—"
                        }
                      </code>
                    </div>
                  </div>


                  <div
                    className={
                      styles.joinMetaGrid
                    }
                  >
                    <div>
                      <span>
                        Clé
                      </span>

                      <strong>
                        {
                          joinKeyLabel(
                            currentIntent
                          )
                        }
                      </strong>
                    </div>

                    <div>
                      <span>
                        Jointure
                      </span>

                      <strong>
                        {
                          joinTypeLabel(
                            currentIntent
                              .join_type
                          )
                        }
                      </strong>
                    </div>

                    <div>
                      <span>
                        Cardinalité attendue
                      </span>

                      <strong>
                        {
                          cardinalityLabel(
                            currentIntent
                              .expected_cardinality
                          )
                        }
                      </strong>
                    </div>

                    <div>
                      <span>
                        Sortie prévue
                      </span>

                      <strong
                        title={
                          currentIntent
                            .output_dataset_id
                        }
                      >
                        {
                          currentIntent
                            .output_dataset_filename
                        }
                      </strong>
                    </div>
                  </div>


                  {
                    warnings.length >
                    0
                      ? (
                          <div
                            className={
                              styles.planEvidence
                            }
                          >
                            <strong>
                              Contrôles du Join Planner
                            </strong>

                            {
                              warnings.map(
                                (
                                  warning,
                                  index
                                ) => (
                                  <p
                                    key={
                                      `${index}-${warning}`
                                    }
                                  >
                                    {
                                      warning
                                    }
                                  </p>
                                )
                              )
                            }
                          </div>
                        )
                      : null
                  }


                  <div
                    className={
                      styles.approvalFooter
                    }
                  >
                    <p>
                      L’approbation autorise uniquement cette proposition
                      exacte. Si le plan serveur change, son
                      <code>
                        request_id
                      </code>
                      devient périmé et l’exécution est refusée.
                    </p>

                    <button
                      type="button"
                      className={
                        styles.primaryButton
                      }
                      onClick={
                        handleApprove
                      }
                      disabled={
                        !canApprove
                      }
                    >
                      {
                        approving
                          ? "Jointure en cours…"
                          : discovery
                              ?.ready_for_approval
                            ? "Approuver la jointure"
                            : "Jointure non approuvable"
                      }
                    </button>
                  </div>
                </article>
              )
            : discovery
                ?.has_candidate ===
                false
              ? (
                  <div
                    className={
                      styles.noCandidate
                    }
                  >
                    <strong>
                      {
                        combine?.status ===
                          "passed"
                          ? "Combinaison terminée"
                          : "Aucune relation sûre à combiner"
                      }
                    </strong>

                    <p>
                      {
                        discovery.reason
                      }
                    </p>

                    {
                      combine?.status ===
                        "passed"
                        ? (
                            <span>
                              Le ou les artefacts combinés sont maintenant
                              disponibles dans l’étape Validation pour
                              sélectionner la sortie analytique finale.
                            </span>
                          )
                        : null
                    }
                  </div>
                )
              : null
        }


        {
          error
            ? (
                <div
                  className={
                    styles.error
                  }
                  role="alert"
                >
                  <strong>
                    Combinaison indisponible
                  </strong>

                  <span>
                    {
                      error
                    }
                  </span>
                </div>
              )
            : null
        }
      </div>


      <div
        className={
          styles.note
        }
      >
        <strong>
          Pourquoi DataLens demande une approbation ?
        </strong>

        <span>
          Une clé commune ne suffit pas à garantir qu’une jointure
          est correcte. Le moteur vérifie notamment la cardinalité
          et les garde-fous avant de proposer l’opération ; l’analyste
          reste responsable de son approbation.
        </span>
      </div>
    </section>
  );
}
