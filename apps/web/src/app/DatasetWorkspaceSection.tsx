import type {
  Dispatch,
  SetStateAction,
} from "react";

import {
  analysisKindLabel,
  formatNumber,
  friendlyVariableLabel,
} from "../components/analysis/analysisPresentation";

import type {
  DatasetManifest,
  MultiDatasetIngestion,
} from "./types";

import styles from "./DatasetWorkspaceSection.module.css";


type DatasetWorkspaceSectionProps = {
  ingestion:
    MultiDatasetIngestion;

  activeManifest:
    DatasetManifest |
    null;

  activeDatasetIndex:
    number;

  setActiveDatasetIndex:
    Dispatch<
      SetStateAction<number>
    >;
};


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


function missingCellCount(
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


export default function DatasetWorkspaceSection({
  ingestion,
  activeManifest,
  activeDatasetIndex,
  setActiveDatasetIndex,
}: DatasetWorkspaceSectionProps) {
  const activeMissingCells =
    activeManifest
      ? missingCellCount(
          activeManifest
        )
      : 0;


  return (
    <section
      className={
        styles.workspace
      }
    >
      <header
        className={
          styles.sectionHeader
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            DATASETS
          </span>

          <h2>
            Sources disponibles
          </h2>

          <p>
            Inspectez la structure détectée avant de poursuivre
            vers la préparation.
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
          styles.datasetGrid
        }
      >
        {
          ingestion.datasets.map(
            (
              manifest,
              index
            ) => {
              const active =
                index ===
                activeDatasetIndex;

              const missingCells =
                missingCellCount(
                  manifest
                );


              return (
                <button
                  className={
                    `${styles.datasetCard} ${
                      active
                        ? styles.datasetCardActive
                        : ""
                    }`
                  }
                  key={
                    manifest.dataset_id
                  }
                  type="button"
                  onClick={
                    () =>
                      setActiveDatasetIndex(
                        index
                      )
                  }
                >
                  <div
                    className={
                      styles.datasetCardTop
                    }
                  >
                    <span
                      className={
                        styles.fileType
                      }
                    >
                      CSV
                    </span>

                    {
                      active
                        ? (
                            <span
                              className={
                                styles.selectedBadge
                              }
                            >
                              <span
                                aria-hidden="true"
                              />

                              Sélectionné
                            </span>
                          )
                        : (
                            <span
                              className={
                                styles.availableBadge
                              }
                            >
                              Disponible
                            </span>
                          )
                    }
                  </div>


                  <strong
                    className={
                      styles.datasetFilename
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
                      styles.datasetCardStats
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
                      <strong>
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
                  </div>
                </button>
              );
            }
          )
        }
      </div>


      {
        activeManifest
          ? (
              <article
                className={
                  styles.profile
                }
              >
                <header
                  className={
                    styles.profileHeader
                  }
                >
                  <div>
                    <span
                      className={
                        styles.eyebrow
                      }
                    >
                      DATASET PROFILE
                    </span>

                    <h3>
                      {
                        activeManifest.filename
                      }
                    </h3>

                    <p>
                      Structure détectée par DataLens lors de
                      l’ingestion locale.
                    </p>
                  </div>


                  <span
                    className={
                      styles.profileFormat
                    }
                  >
                    {
                      activeManifest.extension
                        .replace(
                          ".",
                          ""
                        )
                        .toUpperCase()
                    }
                  </span>
                </header>


                <div
                  className={
                    styles.profileMetrics
                  }
                >
                  <div
                    className={
                      styles.profileMetric
                    }
                  >
                    <strong>
                      {
                        formatNumber(
                          activeManifest.row_count
                        )
                      }
                    </strong>

                    <span>
                      Lignes
                    </span>
                  </div>


                  <div
                    className={
                      styles.profileMetric
                    }
                  >
                    <strong>
                      {
                        activeManifest.column_count
                      }
                    </strong>

                    <span>
                      Colonnes
                    </span>
                  </div>


                  <div
                    className={
                      styles.profileMetric
                    }
                  >
                    <strong>
                      {
                        formatNumber(
                          activeMissingCells
                        )
                      }
                    </strong>

                    <span>
                      Cellules manquantes
                    </span>
                  </div>


                  <div
                    className={
                      styles.profileMetric
                    }
                  >
                    <strong>
                      {
                        formatBytes(
                          activeManifest.memory_bytes
                        )
                      }
                    </strong>

                    <span>
                      Empreinte
                    </span>
                  </div>
                </div>


                <div
                  className={
                    styles.schema
                  }
                >
                  <div
                    className={
                      styles.schemaHeader
                    }
                  >
                    <span>
                      Variable
                    </span>

                    <span>
                      Type
                    </span>

                    <span>
                      Manquantes
                    </span>

                    <span>
                      Distinctes
                    </span>
                  </div>


                  {
                    activeManifest
                      .columns
                      .map(
                        (
                          column
                        ) => (
                          <div
                            className={
                              styles.schemaRow
                            }
                            key={
                              column.name
                            }
                          >
                            <div
                              className={
                                styles.variable
                              }
                            >
                              <strong>
                                {
                                  friendlyVariableLabel(
                                    column.name
                                  )
                                }
                              </strong>

                              <span>
                                {
                                  column.name
                                }
                              </span>
                            </div>


                            <span
                              className={
                                styles.kindBadge
                              }
                            >
                              {
                                analysisKindLabel(
                                  column.analysis_kind
                                )
                              }
                            </span>


                            <span
                              className={
                                column.missing_count >
                                  0
                                  ? styles.missingWarning
                                  : styles.missingNone
                              }
                            >
                              {
                                column.missing_count ===
                                0
                                  ? "Aucune"
                                  : formatNumber(
                                      column.missing_count
                                    )
                              }
                            </span>


                            <span
                              className={
                                styles.numericValue
                              }
                            >
                              {
                                formatNumber(
                                  column.unique_count
                                )
                              }
                            </span>
                          </div>
                        )
                      )
                  }
                </div>


                <footer
                  className={
                    styles.analysisAvailability
                  }
                >
                  <span
                    className={
                      styles.analysisAvailabilityIcon
                    }
                    aria-hidden="true"
                  >
                    ✓
                  </span>

                  <div>
                    <strong>
                      Inclus dans l’analyse
                    </strong>

                    <p>
                      Ce dataset est disponible pour le moteur
                      analytique déterministe de DataLens.
                    </p>
                  </div>


                  <span
                    className={
                      styles.localBadge
                    }
                  >
                    LOCAL
                  </span>
                </footer>
              </article>
            )
          : null
      }
    </section>
  );
}