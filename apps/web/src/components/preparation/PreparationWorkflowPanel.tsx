"use client";


import type {
  PreparationSessionView,
  PreparationStage,
  PreparationStageRecord,
  PreparationStageStatus,
} from "./preparationTypes";

import styles from "./PreparationWorkflowPanel.module.css";


type PreparationWorkflowPanelProps = {
  session:
    PreparationSessionView |
    null;

  loading?:
    boolean;

  error?:
    string |
    null;

  onRefresh?:
    () => void;
};


const STAGE_LABELS:
  Record<
    PreparationStage,
    {
      index:
        string;

      title:
        string;

      description:
        string;
    }
  > = {
    import: {
      index:
        "01",

      title:
        "Import",

      description:
        "Chargement et identification des datasets.",
    },

    understand: {
      index:
        "02",

      title:
        "Compréhension",

      description:
        "Structure, types, rôles et grain des données.",
    },

    quality: {
      index:
        "03",

      title:
        "Qualité",

      description:
        "Valeurs manquantes, doublons et anomalies.",
    },

    clean: {
      index:
        "04",

      title:
        "Nettoyage",

      description:
        "Corrections déterministes et décisions approuvées.",
    },

    transform: {
      index:
        "05",

      title:
        "Transformation",

      description:
        "Variables dérivées, conversions et agrégations.",
    },

    combine: {
      index:
        "06",

      title:
        "Combinaison",

      description:
        "Jointures contrôlées entre datasets.",
    },

    validate: {
      index:
        "07",

      title:
        "Validation",

      description:
        "Contrôle final avant le passage à l’analyse.",
    },
  };


const STATUS_LABELS:
  Record<
    PreparationStageStatus,
    string
  > = {
    not_started:
      "À faire",

    review_required:
      "Revue requise",

    blocked:
      "Bloqué",

    passed:
      "Validé",

    skipped:
      "Non requis",
  };


const STATUS_SYMBOLS:
  Record<
    PreparationStageStatus,
    string
  > = {
    not_started:
      "—",

    review_required:
      "!",

    blocked:
      "×",

    passed:
      "✓",

    skipped:
      "○",
  };


function stageClassName(
  status:
    PreparationStageStatus
): string {
  switch (
    status
  ) {
    case "passed":
      return (
        styles.stagePassed
      );

    case "review_required":
      return (
        styles.stageReview
      );

    case "blocked":
      return (
        styles.stageBlocked
      );

    case "skipped":
      return (
        styles.stageSkipped
      );

    default:
      return (
        styles.stagePending
      );
  }
}


function StageRow({
  record,
}: {
  record:
    PreparationStageRecord;
}) {
  const metadata =
    STAGE_LABELS[
      record.stage
    ];


  return (
    <article
      className={
        `${styles.stage} ${
          stageClassName(
            record.status
          )
        }`
      }
    >
      <div
        className={
          styles.stageIndex
        }
      >
        {
          metadata.index
        }
      </div>


      <div
        className={
          styles.stageBody
        }
      >
        <div
          className={
            styles.stageHeading
          }
        >
          <div>
            <strong>
              {
                metadata.title
              }
            </strong>

            <p>
              {
                metadata.description
              }
            </p>
          </div>


          <div
            className={
              styles.status
            }
          >
            <span
              className={
                styles.statusSymbol
              }
              aria-hidden="true"
            >
              {
                STATUS_SYMBOLS[
                  record.status
                ]
              }
            </span>

            <span>
              {
                STATUS_LABELS[
                  record.status
                ]
              }
            </span>
          </div>
        </div>


        {
          record
            .blocking_reasons
            .length >
          0
            ? (
                <div
                  className={
                    styles.reasons
                  }
                >
                  {
                    record
                      .blocking_reasons
                      .map(
                        (
                          reason
                        ) => (
                          <p
                            key={
                              reason
                            }
                          >
                            {
                              reason
                            }
                          </p>
                        )
                      )
                  }
                </div>
              )
            : null
        }
      </div>
    </article>
  );
}


export default function PreparationWorkflowPanel({
  session,
  loading =
    false,
  error =
    null,
  onRefresh,
}: PreparationWorkflowPanelProps) {
  if (
    loading &&
    !session
  ) {
    return (
      <section
        className={
          styles.panel
        }
      >
        <div
          className={
            styles.loading
          }
        >
          Lecture de l’état de préparation…
        </div>
      </section>
    );
  }


  if (
    !session
  ) {
    return (
      <section
        className={
          styles.panel
        }
      >
        <div
          className={
            styles.empty
          }
        >
          <span
            className={
              styles.eyebrow
            }
          >
            Workflow serveur
          </span>

          <h3>
            Aucune session de préparation
          </h3>

          <p>
            La session sera créée à partir
            des datasets chargés avant
            l’analyse.
          </p>

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
        </div>
      </section>
    );
  }


  const snapshot =
    session.snapshot;


  return (
    <section
      className={
        styles.panel
      }
      aria-labelledby="preparation-workflow-title"
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
            Workflow serveur
          </span>

          <h3
            id="preparation-workflow-title"
          >
            Préparation des données
          </h3>

          <p>
            L’état est calculé par FastAPI.
            L’interface ne peut pas déclarer
            une étape validée elle-même.
          </p>
        </div>


        <div
          className={
            snapshot
              .ready_for_analysis
              ? styles.readinessReady
              : styles.readinessLocked
          }
        >
          <span
            aria-hidden="true"
          >
            {
              snapshot
                .ready_for_analysis
                ? "✓"
                : "×"
            }
          </span>

          <div>
            <small>
              Analyse
            </small>

            <strong>
              {
                snapshot
                  .ready_for_analysis
                  ? "Prête"
                  : "Verrouillée"
              }
            </strong>
          </div>
        </div>
      </header>


      <div
        className={
          styles.meta
        }
      >
        <div>
          <span>
            Session
          </span>

          <strong>
            {
              session.workflow_id
            }
          </strong>
        </div>

        <div>
          <span>
            Révision
          </span>

          <strong>
            {
              session.revision
            }
          </strong>
        </div>

        <div>
          <span>
            Étapes résolues
          </span>

          <strong>
            {
              snapshot
                .resolved_stage_count
            }
            {" / "}
            {
              snapshot
                .stage_count
            }
          </strong>
        </div>

        <div>
          <span>
            Datasets validés
          </span>

          <strong>
            {
              snapshot
                .validated_analysis_dataset_ids
                .length
            }
            {" / "}
            {
              snapshot
                .selected_analysis_dataset_ids
                .length
            }
          </strong>
        </div>
      </div>


      <div
        className={
          styles.stages
        }
      >
        {
          snapshot
            .stages
            .map(
              (
                record
              ) => (
                <StageRow
                  key={
                    record.stage
                  }
                  record={
                    record
                  }
                />
              )
            )
        }
      </div>


      {
        snapshot.next_stage
          ? (
              <div
                className={
                  styles.nextStage
                }
              >
                <span>
                  Prochaine étape
                </span>

                <strong>
                  {
                    STAGE_LABELS[
                      snapshot
                        .next_stage
                    ].title
                  }
                </strong>
              </div>
            )
          : null
      }


      {
        snapshot
          .blocking_reasons
          .length >
        0
          ? (
              <details
                className={
                  styles.blockers
                }
              >
                <summary>
                  Points bloquants
                  {" · "}
                  {
                    snapshot
                      .blocking_reasons
                      .length
                  }
                </summary>

                <div>
                  {
                    snapshot
                      .blocking_reasons
                      .map(
                        (
                          reason
                        ) => (
                          <p
                            key={
                              reason
                            }
                          >
                            {
                              reason
                            }
                          </p>
                        )
                      )
                  }
                </div>
              </details>
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


      {
        onRefresh
          ? (
              <div
                className={
                  styles.actions
                }
              >
                <button
                  type="button"
                  onClick={
                    onRefresh
                  }
                  disabled={
                    loading
                  }
                >
                  {
                    loading
                      ? "Actualisation…"
                      : "Actualiser l’état"
                  }
                </button>
              </div>
            )
          : null
      }
    </section>
  );
}