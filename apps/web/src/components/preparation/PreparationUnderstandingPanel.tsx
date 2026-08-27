import type {
  MultiDatasetIngestion,
} from "../../app/types";

import {
  formatNumber,
} from "../analysis/analysisPresentation";

import styles from "../../app/page.module.css";


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


  return (
    <section
      style={{
        marginTop:
          "18px",

        padding:
          "18px",

        border:
          "1px solid rgba(126, 177, 255, 0.12)",

        borderRadius:
          "16px",

        background:
          "linear-gradient(180deg, rgba(126,177,255,0.032), rgba(255,255,255,0.012))",
      }}
    >
      <div
        style={{
          display:
            "flex",

          alignItems:
            "flex-start",

          justifyContent:
            "space-between",

          gap:
            "18px",

          flexWrap:
            "wrap",
        }}
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Compréhension des données
          </span>

          <h3
            style={{
              margin:
                "7px 0 0",

              fontSize:
                "1.02rem",
            }}
          >
            Structure et périmètre des jeux de données
          </h3>

          <p
            style={{
              margin:
                "7px 0 0",

              maxWidth:
                "800px",

              opacity:
                0.66,

              fontSize:
                "0.79rem",

              lineHeight:
                1.58,
            }}
          >
            Avant de corriger quoi que ce soit, DataLens présente les
            fichiers chargés, leur volume et leur structure. Cette étape
            sert à comprendre le périmètre avant le contrôle de qualité.
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
          styles.metricGrid
        }
        style={{
          marginTop:
            "16px",
        }}
      >
        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Datasets
          </span>

          <strong>
            {
              ingestion.dataset_count
            }
          </strong>
        </article>

        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Lignes totales
          </span>

          <strong>
            {
              formatNumber(
                ingestion.total_rows
              )
            }
          </strong>
        </article>
      </div>


      <div
        style={{
          display:
            "grid",

          gridTemplateColumns:
            "repeat(auto-fit, minmax(190px, 1fr))",

          gap:
            "8px",

          marginTop:
            "12px",
        }}
      >
        {
          ingestion.datasets.map(
            (
              manifest
            ) => (
              <article
                key={
                  manifest.dataset_id
                }
                style={{
                  minWidth:
                    0,

                  padding:
                    "12px",

                  border:
                    "1px solid rgba(255,255,255,0.055)",

                  borderRadius:
                    "10px",

                  background:
                    "rgba(255,255,255,0.01)",
                }}
              >
                <span
                  style={{
                    display:
                      "block",

                    fontSize:
                      "0.54rem",

                    fontWeight:
                      800,

                    letterSpacing:
                      "0.07em",

                    textTransform:
                      "uppercase",

                    opacity:
                      0.42,
                  }}
                >
                  CSV
                </span>

                <strong
                  title={
                    manifest.filename
                  }
                  style={{
                    display:
                      "block",

                    marginTop:
                      "7px",

                    overflow:
                      "hidden",

                    textOverflow:
                      "ellipsis",

                    whiteSpace:
                      "nowrap",

                    fontSize:
                      "0.7rem",
                  }}
                >
                  {
                    manifest.filename
                  }
                </strong>

                <span
                  style={{
                    display:
                      "block",

                    marginTop:
                      "6px",

                    fontSize:
                      "0.59rem",

                    opacity:
                      0.5,
                  }}
                >
                  {
                    formatNumber(
                      manifest.row_count
                    )
                  }
                  {" lignes · "}
                  {
                    manifest.column_count
                  }
                  {" colonnes"}
                </span>
              </article>
            )
          )
        }
      </div>


      <div
        style={{
          marginTop:
            "14px",

          paddingTop:
            "12px",

          borderTop:
            "1px solid rgba(255,255,255,0.06)",

          fontSize:
            "0.64rem",

          lineHeight:
            1.5,

          opacity:
            0.56,
        }}
      >
        Aucune donnée n’est modifiée ici. Le contrôle des anomalies est
        présenté séparément dans l’étape Qualité.
      </div>
    </section>
  );
}
