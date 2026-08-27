import type { Dispatch, SetStateAction } from "react";

import type { DatasetManifest } from "./types";
import type { MultiDatasetIngestion } from "./types";

import { friendlyVariableLabel } from "../components/analysis/analysisPresentation";
import { analysisKindLabel } from "../components/analysis/analysisPresentation";
import { formatNumber } from "../components/analysis/analysisPresentation";
import styles from "./page.module.css";


type DatasetWorkspaceSectionProps = {
  ingestion: MultiDatasetIngestion;
  activeManifest: DatasetManifest | null;
  activeDatasetIndex: number;
  setActiveDatasetIndex: Dispatch<SetStateAction<number>>;
};


export default function DatasetWorkspaceSection({
  ingestion,
  activeManifest,
  activeDatasetIndex,
  setActiveDatasetIndex,
}: DatasetWorkspaceSectionProps) {
  return (
<section
                    className={
                      styles.datasetWorkspace
                    }
                  >
                    <div
                      className={
                        styles.datasetWorkspaceHeader
                      }
                    >
                      <div>
                        <span
                          className={
                            styles.eyebrow
                          }
                        >
                          Datasets chargés
                        </span>

                        <h2>
                          Fichiers détectés
                        </h2>

                        <p>
                          Vérifiez les colonnes et leur typage détecté avant de
                          poursuivre. La préparation détaillée aura lieu à l’étape 3.
                        </p>
                      </div>

                      <span
                        className={
                          styles.sectionStatus
                        }
                      >
                        {
                          ingestion.dataset_count
                        }
                        {" dataset"}
                        {
                          ingestion.dataset_count >
                            1
                            ? "s"
                            : ""
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
                            manifest,
                            index
                          ) => {
                            const active =
                              index ===
                              activeDatasetIndex;


                            return (
                              <button
                                className={
                                  `${styles.datasetTile} ${
                                    active
                                      ? styles.datasetTileActive
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
                                    styles.datasetTileTop
                                  }
                                >
                                  <span
                                    className={
                                      styles.datasetIcon
                                    }
                                  >
                                    CSV
                                  </span>

                                  {
                                    active
                                      ? (
                                          <span
                                            className={
                                              styles.selectedPill
                                            }
                                          >
                                            Sélectionné
                                          </span>
                                        )
                                      : null
                                  }
                                </div>


                                <strong
                                  className={
                                    styles.datasetName
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
                                  <span>
                                    {
                                      formatNumber(
                                        manifest.row_count
                                      )
                                    } lignes
                                  </span>

                                  <span>
                                    {
                                      manifest.column_count
                                    } colonnes
                                  </span>
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
                            <div
                              className={
                                styles.activeDataset
                              }
                            >
                              <div
                                className={
                                  styles.activeDatasetHeader
                                }
                              >
                                <div>
                                  <h3>
                                    {
                                      activeManifest.filename
                                    }
                                  </h3>

                                  <p>
                                    {
                                      formatNumber(
                                        activeManifest.row_count
                                      )
                                    } lignes
                                    {" · "}
                                    {
                                      activeManifest.column_count
                                    } colonnes
                                  </p>
                                </div>
                              </div>


                              <div
                                className={
                                  styles.columnTable
                                }
                              >
                                <div
                                  className={
                                    styles.columnTableHeader
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
                                  activeManifest.columns.map(
                                    (
                                      column
                                    ) => (
                                      <div
                                        className={
                                          styles.columnRow
                                        }
                                        key={
                                          column.name
                                        }
                                      >
                                        <strong>
                                          {
                                            friendlyVariableLabel(
                                              column.name
                                            )
                                          }
                                        </strong>

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

                                        <span>
                                          {
                                            column.missing_count ===
                                            0
                                              ? "Aucune"
                                              : formatNumber(
                                                  column.missing_count
                                                )
                                          }
                                        </span>

                                        <span>
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


                              <div
                                className={
                                  `${styles.availabilityCard} ${styles.availabilityReady}`
                                }
                              >
                                <div>
                                  <strong>
                                    Inclus dans
                                    l’analyse globale
                                  </strong>

                                  <p>
                                    Le moteur Python utilisera
                                    ce fichier pour découvrir
                                    et exécuter les analyses
                                    compatibles.
                                  </p>
                                </div>
                              </div>
                            </div>
                          )
                        : null
                    }
                  </section>
  );
}
