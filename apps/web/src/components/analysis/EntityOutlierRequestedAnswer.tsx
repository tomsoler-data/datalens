import type {
  EntityOutlierFindingEvidenceView,
  EntityOutlierFindingView,
} from "./analysisTypes";

import {
  formatDecimal,
  formatNumber,
} from "./analysisPresentation";

import styles from "../../app/page.module.css";


function entityEvidenceDirectionLabel(
  direction:
    string
): string {
  switch (
    direction
  ) {
    case "high":
      return "Valeur élevée";

    case "low":
      return "Valeur faible";

    default:
      return direction
        .replace(
          /_/g,
          " "
        );
  }
}


function formatEntityEvidenceValue(
  evidence:
    EntityOutlierFindingEvidenceView
): string {
  if (
    evidence.metric ===
      "purchase_sessions" ||
    evidence.metric ===
      "total_items"
  ) {
    return formatNumber(
      Math.round(
        evidence.value
      )
    );
  }


  return formatDecimal(
    evidence.value
  );
}


export default function EntityOutlierRequestedAnswer({
  finding,
  objective,
}: {
  finding:
    EntityOutlierFindingView;

  objective:
    string;
}) {
  const titleId =
    `entity-outlier-answer-${finding.analysis_id.replace(
      /[^a-zA-Z0-9_-]/g,
      "-"
    )}`;


  if (
    finding.status ===
    "blocked"
  ) {
    return (
      <section
        className={
          `${styles.requestedAnswerGroup} ${styles.entityAnswerGroup}`
        }
        aria-labelledby={
          titleId
        }
      >
        <div
          className={
            styles.requestedAnswerGroupHeader
          }
        >
          <div>
            <span
              className={
                styles.eyebrow
              }
            >
              Votre demande
            </span>

            <h2
              id={
                titleId
              }
            >
              Analyse d’entités non exécutable
            </h2>

            <p
              className={
                styles.requestedQuestion
              }
            >
              «
              {
                objective.trim()
              }
              »
            </p>
          </div>

          <span
            className={
              styles.requestedAnswerCount
            }
          >
            Analyse bloquée
          </span>
        </div>


        <section
          className={
            styles.entityAnswerCard
          }
        >
          <div
            className={
              styles.technicalReasons
            }
          >
            <strong>
              DataLens n’invente pas le grain client
            </strong>

            {
              finding.blockers.length >
              0
                ? finding.blockers.map(
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
                : (
                    <p>
                      Aucun contexte analytique client
                      suffisamment sûr n’a pu être
                      construit pour cette demande.
                    </p>
                  )
            }
          </div>
        </section>
      </section>
    );
  }


  const priorityProfiles =
    finding.priority_profiles
      .slice(
        0,
        8
      );


  return (
    <section
      className={
        `${styles.requestedAnswerGroup} ${styles.entityAnswerGroup}`
      }
      aria-labelledby={
        titleId
      }
    >
      <div
        className={
          styles.requestedAnswerGroupHeader
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Votre demande
          </span>

          <h2
            id={
              titleId
            }
          >
            Réponse à votre demande
          </h2>

          <p
            className={
              styles.requestedQuestion
            }
          >
            «
            {
              objective.trim()
            }
            »
          </p>
        </div>


        <span
          className={
            styles.requestedAnswerCount
          }
        >
          {
            finding.priority_profile_count
          }
          {" profil"}
          {
            finding.priority_profile_count >
            1
              ? "s"
              : ""
          }
          {" prioritaire"}
          {
            finding.priority_profile_count >
            1
              ? "s"
              : ""
          }
        </span>
      </div>


      <section
        className={
          styles.entityAnswerCard
        }
      >
        <div
          className={
            styles.entityAnswerLead
          }
        >
          <div>
            <span
              className={
                styles.eyebrow
              }
            >
              Résultat vérifié · grain client
            </span>

            <h3
              className={
                styles.entityAnswerTitle
              }
            >
              {
                finding.priority_profile_count ===
                  0
                  ? "Aucun profil extrêmement atypique détecté"
                  : finding.priority_profile_count ===
                      1
                    ? "1 profil client nécessite une revue prioritaire"
                    : `${formatNumber(
                        finding.priority_profile_count
                      )} profils clients nécessitent une revue prioritaire`
              }
            </h3>

            <p
              className={
                styles.entityAnswerDescription
              }
            >
              {
                formatNumber(
                  finding.entity_count
                )
              }
              {" clients analysés · "}
              {
                finding.dataset_filename ??
                "vue analytique client"
              }
            </p>
          </div>


          <div
            className={
              styles.entityAnswerStats
            }
          >
            <article>
              <span>
                Clients analysés
              </span>

              <strong>
                {
                  formatNumber(
                    finding.entity_count
                  )
                }
              </strong>
            </article>

            <article>
              <span>
                Profils prioritaires
              </span>

              <strong>
                {
                  formatNumber(
                    finding.priority_profile_count
                  )
                }
              </strong>
            </article>

            <article>
              <span>
                Signaux secondaires
              </span>

              <strong>
                {
                  formatNumber(
                    finding.behavioral_signal_count
                  )
                }
              </strong>
            </article>
          </div>
        </div>


        {
          priorityProfiles.length >
          0
            ? (
                <div
                  className={
                    styles.entityProfileGrid
                  }
                >
                  {
                    priorityProfiles.map(
                      (
                        profile,
                        profileIndex
                      ) => (
                        <article
                          className={
                            styles.entityProfileCard
                          }
                          key={
                            profile.entity
                          }
                        >
                          <header
                            className={
                              styles.entityProfileHeader
                            }
                          >
                            <div>
                              <span
                                className={
                                  styles.entityProfileIndex
                                }
                              >
                                Profil
                                {" "}
                                {
                                  String(
                                    profileIndex +
                                    1
                                  ).padStart(
                                    2,
                                    "0"
                                  )
                                }
                              </span>

                              <h3>
                                {
                                  profile.entity
                                }
                              </h3>

                              <p>
                                {
                                  profile.dominant_family_label
                                }
                              </p>
                            </div>

                            <span
                              className={
                                styles.entitySignalBadge
                              }
                            >
                              {
                                profile.signal_count
                              }
                              {" signal"}
                              {
                                profile.signal_count >
                                1
                                  ? "s"
                                  : ""
                              }
                            </span>
                          </header>


                          <div
                            className={
                              styles.entityMetricGrid
                            }
                          >
                            {
                              profile.evidence
                                .slice(
                                  0,
                                  3
                                )
                                .map(
                                  (
                                    evidence
                                  ) => (
                                    <div
                                      className={
                                        styles.entityMetric
                                      }
                                      key={
                                        `${profile.entity}-${evidence.metric}`
                                      }
                                    >
                                      <span>
                                        {
                                          evidence.metric_label
                                        }
                                      </span>

                                      <strong>
                                        {
                                          formatEntityEvidenceValue(
                                            evidence
                                          )
                                        }
                                      </strong>
                                    </div>
                                  )
                                )
                            }
                          </div>


                          <details
                            className={
                              styles.entityProfileDetails
                            }
                          >
                            <summary>
                              Voir pourquoi ce profil ressort
                            </summary>

                            <div
                              className={
                                styles.entityProfileEvidence
                              }
                            >
                              <p>
                                {
                                  profile.explanation
                                }
                              </p>

                              {
                                profile.evidence.map(
                                  (
                                    evidence
                                  ) => (
                                    <div
                                      key={
                                        `${profile.entity}-${evidence.metric}-detail`
                                      }
                                    >
                                      <span>
                                        {
                                          evidence.metric_label
                                        }
                                      </span>

                                      <strong>
                                        {
                                          entityEvidenceDirectionLabel(
                                            evidence.direction
                                          )
                                        }
                                        {" · "}
                                        {
                                          formatDecimal(
                                            evidence.distance_iqr
                                          )
                                        }
                                        {" IQR au-delà du seuil"}
                                      </strong>
                                    </div>
                                  )
                                )
                              }
                            </div>
                          </details>
                        </article>
                      )
                    )
                  }
                </div>
              )
            : null
        }


        <div
          className={
            styles.entityAnswerFooter
          }
        >
          {
            finding.behavioral_signal_count >
            0
              ? (
                  <details
                    className={
                      styles.entitySecondaryDetails
                    }
                  >
                    <summary>
                      <span>
                        Autres signaux comportementaux
                      </span>

                      <strong>
                        {
                          formatNumber(
                            finding.behavioral_signal_count
                          )
                        }
                      </strong>
                    </summary>

                    <div>
                      <p>
                        Ces clients franchissent au moins une
                        borne statistique, mais ne présentent pas
                        le même niveau d’atypicité globale que les
                        profils prioritaires affichés ci-dessus.
                      </p>

                      <p>
                        Ils restent des pistes d’exploration.
                        DataLens ne les classe pas automatiquement
                        comme profils extrêmes.
                      </p>
                    </div>
                  </details>
                )
              : null
          }


          <details
            className={
              styles.entityMethodDetails
            }
          >
            <summary>
              Preuves & méthodologie
            </summary>

            <div
              className={
                styles.entityMethodBody
              }
            >
              {
                finding.methodology.length >
                0
                  ? (
                      <section>
                        <strong>
                          Méthodologie
                        </strong>

                        {
                          finding.methodology.map(
                            (
                              item
                            ) => (
                              <p
                                key={
                                  item
                                }
                              >
                                {
                                  item
                                }
                              </p>
                            )
                          )
                        }
                      </section>
                    )
                  : null
              }


              {
                finding.caveats.length >
                0
                  ? (
                      <section>
                        <strong>
                          Précautions
                        </strong>

                        {
                          finding.caveats.map(
                            (
                              caveat
                            ) => (
                              <p
                                key={
                                  caveat
                                }
                              >
                                {
                                  caveat
                                }
                              </p>
                            )
                          )
                        }
                      </section>
                    )
                  : null
              }


              <div
                className={
                  styles.entityTraceGrid
                }
              >
                <article>
                  <span>
                    Grain
                  </span>

                  <strong>
                    Client
                  </strong>

                  <small>
                    {
                      finding.entity_column ??
                      "Entité dérivée"
                    }
                  </small>
                </article>

                <article>
                  <span>
                    Signaux IQR bruts
                  </span>

                  <strong>
                    {
                      formatNumber(
                        finding.raw_flagged_entity_count
                      )
                    }
                  </strong>

                  <small>
                    Au moins une borne franchie
                  </small>
                </article>

                <article>
                  <span>
                    Adaptation produit
                  </span>

                  <strong>
                    Vérifiée
                  </strong>

                  <small>
                    {
                      finding.adapter_rule_version
                    }
                  </small>
                </article>
              </div>
            </div>
          </details>
        </div>
      </section>
    </section>
  );
}
