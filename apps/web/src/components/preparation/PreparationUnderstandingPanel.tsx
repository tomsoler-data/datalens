import type {
  DatasetManifest,
  MultiDatasetIngestion,
} from "../../app/types";

import {
  formatNumber,
} from "../analysis/analysisPresentation";

import styles
  from "./PreparationUnderstandingPanel.module.css";


function formatBytes(
  bytes:
    number
): string {
  if (
    !Number.isFinite(
      bytes
    ) ||
    bytes <= 0
  ) {
    return "—";
  }


  if (
    bytes <
    1024
  ) {
    return `${bytes} B`;
  }


  if (
    bytes <
    1024 * 1024
  ) {
    return `${(
      bytes /
      1024
    ).toFixed(
      1
    )} KB`;
  }


  return `${(
    bytes /
    (
      1024 *
      1024
    )
  ).toFixed(
    1
  )} MB`;
}


function datasetMissingCells(
  manifest:
    DatasetManifest
): number {
  return manifest
    .columns
    .reduce(
      (
        total,
        column
      ) =>
        total +
        column.missing_count,
      0
    );
}


export default function PreparationUnderstandingPanel({
  ingestion,
}: {
  ingestion:
    MultiDatasetIngestion |
    null;
}) {
  if (
    ingestion ===
    null
  ) {
    return null;
  }


  const totalColumns =
    ingestion.datasets.reduce(
      (
        total,
        dataset
      ) =>
        total +
        dataset.column_count,
      0
    );


  const totalMissingCells =
    ingestion.datasets.reduce(
      (
        total,
        dataset
      ) =>
        total +
        datasetMissingCells(
          dataset
        ),
      0
    );


  const totalMemoryBytes =
    ingestion.datasets.reduce(
      (
        total,
        dataset
      ) =>
        total +
        dataset.memory_bytes,
      0
    );


  const datasetWarningCount =
    ingestion.datasets.reduce(
      (
        total,
        dataset
      ) =>
        total +
        dataset.warnings.length,
      0
    );


  const warningCount =
    ingestion.warnings.length +
    datasetWarningCount;


  return (
    <section
      className={
        styles.panel
      }
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
            DATA PROFILE
          </span>

          <h3>
            Comprendre le périmètre
          </h3>

          <p>
            Avant toute correction, DataLens décrit les sources,
            leur volume et leur structure afin de comprendre le
            contexte analytique sans modifier les données.
          </p>
        </div>


        <span
          className={
            styles.datasetCount
          }
        >
          {
            ingestion.dataset_count
          }

          {
            ingestion.dataset_count ===
            1
              ? " dataset"
              : " datasets"
          }
        </span>
      </header>


      <div
        className={
          styles.metrics
        }
      >
        <article
          className={
            styles.metric
          }
        >
          <span
            className={
              styles.metricIcon
            }
            aria-hidden="true"
          >
            D
          </span>

          <div>
            <strong>
              {
                ingestion.dataset_count
              }
            </strong>

            <span>
              Datasets
            </span>
          </div>
        </article>


        <article
          className={
            styles.metric
          }
        >
          <span
            className={
              styles.metricIcon
            }
            aria-hidden="true"
          >
            R
          </span>

          <div>
            <strong>
              {
                formatNumber(
                  ingestion.total_rows
                )
              }
            </strong>

            <span>
              Lignes
            </span>
          </div>
        </article>


        <article
          className={
            styles.metric
          }
        >
          <span
            className={
              styles.metricIcon
            }
            aria-hidden="true"
          >
            C
          </span>

          <div>
            <strong>
              {
                formatNumber(
                  totalColumns
                )
              }
            </strong>

            <span>
              Colonnes
            </span>
          </div>
        </article>


        <article
          className={
            `${styles.metric} ${
              totalMissingCells > 0
                ? styles.metricAttention
                : ""
            }`
          }
        >
          <span
            className={
              styles.metricIcon
            }
            aria-hidden="true"
          >
            ∅
          </span>

          <div>
            <strong>
              {
                formatNumber(
                  totalMissingCells
                )
              }
            </strong>

            <span>
              Manquantes
            </span>
          </div>
        </article>


        <article
          className={
            styles.metric
          }
        >
          <span
            className={
              styles.metricIcon
            }
            aria-hidden="true"
          >
            M
          </span>

          <div>
            <strong>
              {
                formatBytes(
                  totalMemoryBytes
                )
              }
            </strong>

            <span>
              Empreinte
            </span>
          </div>
        </article>
      </div>


      <div
        className={
          styles.sourcesHeader
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            SOURCES
          </span>

          <strong>
            Jeux de données observés
          </strong>
        </div>


        <span
          className={
            warningCount > 0
              ? styles.warningBadge
              : styles.cleanBadge
          }
        >
          {
            warningCount > 0
              ? `${warningCount} signal${
                  warningCount > 1
                    ? "s"
                    : ""
                }`
              : "Aucun signal d’ingestion"
          }
        </span>
      </div>


      <div
        className={
          styles.datasetGrid
        }
      >
        {
          ingestion.datasets.map(
            (
              manifest
            ) => {
              const missingCells =
                datasetMissingCells(
                  manifest
                );


              return (
                <article
                  className={
                    styles.datasetCard
                  }
                  key={
                    manifest.dataset_id
                  }
                >
                  <div
                    className={
                      styles.datasetTop
                    }
                  >
                    <span
                      className={
                        styles.fileType
                      }
                    >
                      {
                        manifest.extension
                          .replace(
                            ".",
                            ""
                          )
                          .toUpperCase()
                      }
                    </span>

                    <span
                      className={
                        styles.localSignal
                      }
                    >
                      <span
                        aria-hidden="true"
                      />

                      LOCAL
                    </span>
                  </div>


                  <strong
                    className={
                      styles.filename
                    }
                    title={
                      manifest.filename
                    }
                  >
                    {
                      manifest.filename
                    }
                  </strong>


                  <div
                    className={
                      styles.datasetStats
                    }
                  >
                    <div>
                      <strong>
                        {
                          formatNumber(
                            manifest.row_count
                          )
                        }
                      </strong>

                      <span>
                        lignes
                      </span>
                    </div>


                    <div>
                      <strong>
                        {
                          manifest.column_count
                        }
                      </strong>

                      <span>
                        colonnes
                      </span>
                    </div>


                    <div>
                      <strong
                        className={
                          missingCells >
                            0
                            ? styles.attentionValue
                            : undefined
                        }
                      >
                        {
                          formatNumber(
                            missingCells
                          )
                        }
                      </strong>

                      <span>
                        manquantes
                      </span>
                    </div>


                    <div>
                      <strong>
                        {
                          formatBytes(
                            manifest.memory_bytes
                          )
                        }
                      </strong>

                      <span>
                        mémoire
                      </span>
                    </div>
                  </div>


                  {
                    manifest.warnings.length >
                    0
                      ? (
                          <div
                            className={
                              styles.datasetWarning
                            }
                          >
                            <span
                              aria-hidden="true"
                            >
                              !
                            </span>

                            <div>
                              <strong>
                                Signal d’ingestion
                              </strong>

                              <p>
                                {
                                  manifest
                                    .warnings[
                                    0
                                  ]
                                }
                              </p>
                            </div>
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


      <footer
        className={
          styles.observationNotice
        }
      >
        <span
          className={
            styles.observationIcon
          }
          aria-hidden="true"
        >
          ◉
        </span>

        <div>
          <strong>
            Observation uniquement
          </strong>

          <p>
            Aucune donnée n’est modifiée pendant cette étape.
            Les anomalies et décisions de correction sont traitées
            séparément dans le contrôle Qualité.
          </p>
        </div>


        <span
          className={
            styles.ruleVersion
          }
          title={
            ingestion.ingestion_rule_version
          }
        >
          {
            ingestion.ingestion_rule_version
          }
        </span>
      </footer>
    </section>
  );
}