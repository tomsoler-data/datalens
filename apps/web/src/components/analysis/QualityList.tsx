"use client";

import type { ReportQualityItem }
  from "../../app/types";

import {
  formatNumber,
  formatPercent,
} from "./analysisPresentation";

import styles
  from "../../app/page.module.css";


export default function QualityList({
  items,
}: {
  items:
    ReportQualityItem[];
}) {
  if (
    items.length ===
    0
  ) {
    return null;
  }


  return (
    <details
      className={
        styles.technicalPanel
      }
    >
      <summary>
        Qualité des données
        {" · "}
        {
          items.length
        }
      </summary>


      <div
        className={
          styles.explanationGrid
        }
      >
        {
          items.map(
            (
              item
            ) => (
              <article
                className={
                  styles.explanationCard
                }
                key={
                  item.analysis_id
                }
              >
                <span
                  className={
                    styles.eyebrow
                  }
                >
                  Contrôle qualité
                </span>

                <h3
                  className={
                    styles.explanationTitle
                  }
                >
                  {
                    item.dataset
                  }
                </h3>


                <div
                  className={
                    styles.technicalReasons
                  }
                >
                  <p>
                    {
                      formatNumber(
                        item.row_count
                      )
                    } lignes
                    {" · "}
                    {
                      formatNumber(
                        item.column_count
                      )
                    } colonnes
                  </p>

                  <p>
                    Valeurs manquantes :
                    {" "}
                    {
                      formatNumber(
                        item.missing_cells
                      )
                    }
                    {" · "}
                    {
                      formatPercent(
                        item.missing_ratio
                      )
                    }
                  </p>

                  <p>
                    Doublons stricts :
                    {" "}
                    {
                      formatNumber(
                        item.duplicate_rows
                      )
                    }
                    {" · "}
                    {
                      formatPercent(
                        item.duplicate_ratio
                      )
                    }
                  </p>


                  {
                    item
                      .completely_missing_columns
                      .length >
                    0
                      ? (
                          <p>
                            Colonnes entièrement
                            manquantes :
                            {" "}
                            {
                              item
                                .completely_missing_columns
                                .join(
                                  ", "
                                )
                            }
                          </p>
                        )
                      : null
                  }


                  {
                    item
                      .constant_columns
                      .length >
                    0
                      ? (
                          <p>
                            Colonnes constantes :
                            {" "}
                            {
                              item
                                .constant_columns
                                .join(
                                  ", "
                                )
                            }
                          </p>
                        )
                      : null
                  }
                </div>


                {
                  item
                    .summary
                    .map(
                      (
                        line
                      ) => (
                        <p
                          className={
                            styles.explanationText
                          }
                          key={
                            line
                          }
                        >
                          {
                            line
                          }
                        </p>
                      )
                    )
                }
              </article>
            )
          )
        }
      </div>
    </details>
  );
}
