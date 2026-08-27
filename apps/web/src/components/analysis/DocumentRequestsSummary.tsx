"use client";

import type {
  DocumentSummaryView,
  RequestedPlanItemView,
  RequestedPlanView,
} from "./analysisTypes";

import styles
  from "../../app/page.module.css";


function requestPlanningLabel(
  status:
    RequestedPlanItemView[
      "status"
    ]
): string {
  switch (
    status
  ) {
    case "ready":
      return "Prête";

    case "blocked":
      return "Bloquée";

    case "ambiguous":
      return "À clarifier";

    default:
      return status;
  }
}


function requestResolutionGuidance(
  request:
    RequestedPlanItemView
): {
  title:
    string;

  action:
    string;

  protection:
    string;
} {
  if (
    request.kind ===
    "b2b_revenue_share"
  ) {
    return {
      title:
        "Identifier explicitement les clients BtoB",

      action:
        (
          "Ajoutez ou indiquez une colonne qui décrit explicitement " +
          "le type de client, par exemple segment, customer_type, " +
          "account_type ou b2b_flag."
        ),

      protection:
        (
          "DataLens ne déduira pas qu’un client est BtoB à partir " +
          "d’un chiffre d’affaires élevé, d’un panier atypique ou " +
          "d’une fréquence d’achat importante."
        ),
    };
  }


  if (
    request.status ===
    "ambiguous"
  ) {
    return {
      title:
        "Préciser la règle d’analyse",

      action:
        (
          "Précisez la métrique, le périmètre ou la règle attendue " +
          "afin que DataLens puisse construire un plan déterministe."
        ),

      protection:
        (
          "DataLens préfère demander une clarification plutôt que " +
          "choisir arbitrairement une définition qui pourrait changer " +
          "le résultat."
        ),
    };
  }


  return {
    title:
      "Fournir l’information manquante",

    action:
      (
        "Ajoutez la variable explicitement requise dans les données " +
        "ou complétez la documentation afin que la demande puisse " +
        "être résolue sans hypothèse cachée."
      ),

    protection:
      (
        "Aucune substitution approximative n’est exécutée lorsque " +
        "la preuve nécessaire à l’analyse manque."
      ),
  };
}


function RequestResolutionPanel({
  plan,
}: {
  plan:
    RequestedPlanView |
    null;
}) {
  const unresolved =
    plan?.requests.filter(
      (
        request
      ) =>
        request.status !==
        "ready"
    ) ??
    [];


  if (
    unresolved.length ===
    0
  ) {
    return null;
  }


  return (
    <section
      className={
        styles.summaryPanel
      }
    >
      <div
        className={
          styles.sectionHead
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Action requise
          </span>

          <h3>
            {
              unresolved.length ===
                1
                ? "Une demande ne peut pas encore être exécutée"
                : `${unresolved.length} demandes nécessitent votre intervention`
            }
          </h3>

          <p
            className={
              styles.resultSubtitle
            }
          >
            DataLens s’arrête lorsqu’une information
            indispensable manque ou lorsqu’une règle
            reste ambiguë. Le moteur n’invente pas
            la donnée manquante.
          </p>
        </div>
      </div>


      <div
        className={
          styles.explanationGrid
        }
      >
        {
          unresolved.map(
            (
              request
            ) => {
              const guidance =
                requestResolutionGuidance(
                  request
                );


              return (
                <article
                  className={
                    styles.explanationCard
                  }
                  key={
                    request.request_id
                  }
                >
                  <span
                    className={
                      styles.eyebrow
                    }
                  >
                    {
                      request.status ===
                        "blocked"
                        ? "Bloquée"
                        : "À clarifier"
                    }
                  </span>

                  <h3
                    className={
                      styles.explanationTitle
                    }
                  >
                    {
                      request.request_text
                    }
                  </h3>


                  <p
                    className={
                      styles.resultSubtitle
                    }
                  >
                    {
                      request.source_filename
                    }
                    {" · "}
                    {
                      request.source_locator
                    }
                  </p>


                  {
                    request.blockers.length >
                    0
                      ? (
                          <div
                            className={
                              styles.technicalReasons
                            }
                          >
                            <strong>
                              Pourquoi DataLens s’arrête ici
                            </strong>

                            {
                              request.blockers.map(
                                (
                                  blocker
                                ) => (
                                  <p
                                    key={
                                      blocker
                                    }
                                  >
                                    {
                                      blocker
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
                      styles.technicalReasons
                    }
                  >
                    <strong>
                      {
                        guidance.title
                      }
                    </strong>

                    <p>
                      {
                        guidance.action
                      }
                    </p>
                  </div>


                  <p
                    className={
                      styles.explanationText
                    }
                  >
                    {
                      guidance.protection
                    }
                  </p>
                </article>
              );
            }
          )
        }
      </div>
    </section>
  );
}


export default function DocumentRequestsSummary({
  summary,
  plan,
}: {
  summary:
    DocumentSummaryView |
    null;

  plan:
    RequestedPlanView |
    null;
}) {
  if (
    !summary ||
    summary.status !==
      "ready"
  ) {
    return null;
  }


  const summaryPoints =
    summary.summary_points ??
    [];

  const documents =
    summary.documents ??
    [];

  const requestCount =
    plan?.request_count ??
    summary.analytical_request_count;

  const readyCount =
    plan?.ready_count ??
    0;

  const blockedCount =
    plan?.blocked_count ??
    0;

  const ambiguousCount =
    plan?.ambiguous_count ??
    0;

  const clarificationCount =
    blockedCount +
    ambiguousCount;


  return (
    <>
      <div
        className={
          styles.sectionHead
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Documentation métier
          </span>

          <h2>
            Ce que demandent
            vos documents
          </h2>

          <p
            className={
              styles.resultSubtitle
            }
          >
            DataLens distingue le cadrage
            documentaire des résultats calculés.
            Une demande détectée n’est pas
            considérée comme exécutée tant que
            le moteur Python ne l’a pas réellement
            analysée.
          </p>
        </div>
      </div>


      <div
        className={
          styles.metricGrid
        }
      >
        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Documents
          </span>

          <strong>
            {
              summary.document_count
            }
          </strong>
        </article>


        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Demandes détectées
          </span>

          <strong>
            {
              requestCount
            }
          </strong>
        </article>


        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Prêtes à analyser
          </span>

          <strong
            className={
              styles.statusGood
            }
          >
            {
              readyCount
            }
          </strong>
        </article>


        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Intervention requise
          </span>

          <strong
            className={
              clarificationCount >
              0
                ? styles.statusNeutral
                : styles.statusNeutral
            }
          >
            {
              clarificationCount
            }
          </strong>
        </article>
      </div>


      <RequestResolutionPanel
        plan={
          plan
        }
      />


      {
        summaryPoints.length >
        0
          ? (
              <section
                className={
                  styles.summaryPanel
                }
              >
                <div
                  className={
                    styles.summaryItem
                  }
                >
                  <span>
                    Points de cadrage
                  </span>

                  {
                    summaryPoints
                      .slice(
                        0,
                        5
                      )
                      .map(
                        (
                          point
                        ) => (
                          <p
                            key={
                              `${
                                point.citation.chunk_id
                              }-${
                                point.evidence_unit_id
                              }`
                            }
                          >
                            {
                              point.statement
                            }
                          </p>
                        )
                      )
                  }
                </div>
              </section>
            )
          : null
      }


      {
        documents.length >
        0
          ? (
              <div
                className={
                  styles.explanationGrid
                }
              >
                {
                  documents.map(
                    (
                      document
                    ) => {
                      const visiblePoints =
                        document
                          .summary_points
                          .length >
                        0
                          ? document
                              .summary_points
                              .slice(
                                0,
                                2
                              )
                          : document
                              .analytical_requests
                              .slice(
                                0,
                                2
                              );


                      return (
                        <article
                          className={
                            styles.explanationCard
                          }
                          key={
                            document.document_id
                          }
                        >
                          <span
                            className={
                              styles.eyebrow
                            }
                          >
                            Document vérifié
                          </span>

                          <h3
                            className={
                              styles.explanationTitle
                            }
                          >
                            {
                              document.filename
                            }
                          </h3>

                          <p
                            className={
                              styles.resultSubtitle
                            }
                          >
                            {
                              document
                                .analytical_requests
                                .length
                            }
                            {" "}
                            demande
                            {
                              document
                                .analytical_requests
                                .length >
                              1
                                ? "s"
                                : ""
                            }
                            {" · "}
                            {
                              document
                                .verified_claim_count
                            }
                            {" "}
                            élément
                            {
                              document
                                .verified_claim_count >
                              1
                                ? "s"
                                : ""
                            }
                            {" "}
                            vérifié
                            {
                              document
                                .verified_claim_count >
                              1
                                ? "s"
                                : ""
                            }
                          </p>


                          {
                            visiblePoints.map(
                              (
                                point
                              ) => (
                                <p
                                  className={
                                    styles.explanationText
                                  }
                                  key={
                                    `${
                                      document.document_id
                                    }-${
                                      point.evidence_unit_id
                                    }`
                                  }
                                >
                                  {
                                    point.statement
                                  }
                                </p>
                              )
                            )
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
        plan &&
        plan.requests.length >
        0
          ? (
              <details
                className={
                  styles.technicalPanel
                }
              >
                <summary>
                  Voir les
                  {" "}
                  {
                    plan.request_count
                  }
                  {" "}
                  demandes détectées
                </summary>

                <div
                  className={
                    styles.explanationGrid
                  }
                >
                  {
                    plan.requests.map(
                      (
                        request,
                        index
                      ) => (
                        <article
                          className={
                            styles.explanationCard
                          }
                          key={
                            request.request_id
                          }
                        >
                          <span
                            className={
                              styles.eyebrow
                            }
                          >
                            Demande
                            {" "}
                            {
                              String(
                                index + 1
                              ).padStart(
                                2,
                                "0"
                              )
                            }
                            {" · "}
                            {
                              requestPlanningLabel(
                                request.status
                              )
                            }
                          </span>

                          <h3
                            className={
                              styles.explanationTitle
                            }
                          >
                            {
                              request.request_text
                            }
                          </h3>

                          <p
                            className={
                              styles.resultSubtitle
                            }
                          >
                            {
                              request.source_filename
                            }
                            {" · "}
                            {
                              request.source_locator
                            }
                          </p>


                          {
                            request.blockers.length >
                            0
                              ? (
                                  <div
                                    className={
                                      styles.technicalReasons
                                    }
                                  >
                                    {
                                      request.blockers.map(
                                        (
                                          blocker
                                        ) => (
                                          <p
                                            key={
                                              blocker
                                            }
                                          >
                                            {
                                              blocker
                                            }
                                          </p>
                                        )
                                      )
                                    }
                                  </div>
                                )
                              : null
                          }
                        </article>
                      )
                    )
                  }
                </div>
              </details>
            )
          : null
      }
    </>
  );
}
