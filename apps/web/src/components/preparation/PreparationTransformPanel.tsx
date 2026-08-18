"use client";

import type {
  PreparationSessionView,
  PreparationStageRecord,
} from "./preparationTypes";

import styles from "./PreparationTransformPanel.module.css";


type PreparationTransformPanelProps = {
  session: PreparationSessionView | null;
};


function findStage(
  session: PreparationSessionView | null,
  stageName: PreparationStageRecord["stage"]
): PreparationStageRecord | null {
  if (
    session ===
    null
  ) {
    return null;
  }


  return (
    session.snapshot.stages.find(
      (
        stage
      ) =>
        stage.stage ===
        stageName
    ) ??
    null
  );
}


function statusLabel(
  stage: PreparationStageRecord | null
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
  stage: PreparationStageRecord | null
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


export default function PreparationTransformPanel({
  session,
}: PreparationTransformPanelProps) {
  if (
    session ===
    null
  ) {
    return null;
  }


  const transform =
    findStage(
      session,
      "transform"
    );

  const combine =
    findStage(
      session,
      "combine"
    );


  const allSkipped =
    transform?.status ===
      "skipped" &&
    combine?.status ===
      "skipped";


  return (
    <section
      className={
        styles.panel
      }
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

          <h3>
            Transformer et combiner
          </h3>

          <p>
            Cette étape regroupe les transformations déterministes,
            agrégations, variables dérivées et combinaisons de datasets
            qui peuvent être nécessaires avant l’analyse.
          </p>
        </div>

        <span
          className={
            `${styles.badge} ${
              allSkipped
                ? styles.skipped
                : ""
            }`
          }
        >
          {
            allSkipped
              ? "NON REQUIS"
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
            exécutées uniquement par les moteurs Python déterministes.
          </p>

          {
            transform?.blocking_reasons.length
              ? (
                  <div
                    className={
                      styles.reasons
                    }
                  >
                    {
                      transform.blocking_reasons.map(
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
                statusLabel(
                  combine
                )
              }
            </span>
          </div>

          <p>
            Les jointures restent séparées des transformations et doivent
            respecter le grain, les cardinalités et les garde-fous du moteur.
          </p>

          {
            combine?.blocking_reasons.length
              ? (
                  <div
                    className={
                      styles.reasons
                    }
                  >
                    {
                      combine.blocking_reasons.map(
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
          styles.note
        }
      >
        <strong>
          Pourquoi cette étape peut être « Non requis » ?
        </strong>

        <span>
          DataLens ne transforme pas les données pour remplir artificiellement
          le workflow. Si aucune opération structurelle n’est nécessaire pour
          le périmètre validé, les étapes restent explicitement ignorées.
        </span>
      </div>
    </section>
  );
}
