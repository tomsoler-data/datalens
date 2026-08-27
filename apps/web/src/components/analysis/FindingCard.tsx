"use client";

import ExpandableChart from "./charts/ExpandableChart";
import FindingChart from "./charts/FindingChart";
import type { FindingRagContext } from "../../app/types";
import type { ReportFinding } from "../../app/types";
import { datumNumber } from "./analysisPresentation";
import { familyLabel } from "./analysisPlanningPresentation";
import { lineBandRenderablePoints } from "./charts/LineBandChart";
import styles from "../../app/page.module.css";


function RagContextBlock({
  context,
}: {
  context:
    FindingRagContext |
    null;
}) {
  if (
    !context
  ) {
    return null;
  }


  if (
    context.explanation.status ===
      "ready" &&
    context.explanation.claims.length >
      0
  ) {
    return (
      <details
        className={
          styles.technicalPanel
        }
      >
        <summary>
          Contexte documentaire
          {" · "}
          vérifié
        </summary>


        <div
          className={
            styles.technicalReasons
          }
        >
          <p>
            {
              context
                .explanation
                .explanation
            }
          </p>
        </div>


        <div
          className={
            styles.evidenceFlow
          }
        >
          {
            context
              .explanation
              .claims
              .map(
                (
                  claim,
                  index
                ) => (
                  <article
                    className={
                      styles.evidenceItem
                    }
                    key={
                      `${
                        claim
                          .citation
                          .chunk_id
                      }-${index}`
                    }
                  >
                    <span>
                      Preuve documentaire
                    </span>

                    <strong>
                      {
                        claim
                          .citation
                          .filename
                      }
                    </strong>

                    <small>
                      {
                        claim
                          .evidence_quote
                      }
                    </small>

                    <small>
                      {
                        claim
                          .citation
                          .source_locator
                      }
                    </small>
                  </article>
                )
              )
          }
        </div>


        <div
          className={
            styles.technicalReasons
          }
        >
          <p>
            Relation documentaire :
            {" "}
            {
              context
                .documentary_context
                .relation_type ??
              "non spécifiée"
            }
          </p>

          <p>
            Force :
            {" "}
            {
              context
                .documentary_context
                .strength ??
              "non spécifiée"
            }
          </p>
        </div>
      </details>
    );
  }


  return (
    <details
      className={
        styles.technicalPanel
      }
    >
      <summary>
        Contexte documentaire
        {" · "}
        non retenu
      </summary>

      <div
        className={
          styles.technicalReasons
        }
      >
        <p>
          {
            context
              .explanation
              .abstention_reason ??
            context
              .abstention_reason ??
            "Aucune preuve documentaire suffisamment directe n’a été retenue."
          }
        </p>
      </div>
    </details>
  );
}


function findingHasDetailedVisualization(
  finding:
    ReportFinding
): boolean {
  const data =
    finding.chart_data ??
    [];


  switch (
    finding.chart_type
  ) {
    case "line":
      return (
        data.filter(
          (
            datum
          ) =>
            datumNumber(
              datum,
              "value"
            ) !==
            null
        ).length >
        1
      );


    case "line_band":
      return (
        lineBandRenderablePoints(
          data
        ).length >
        1
      );


    case "scatter":
      return (
        data.filter(
          (
            datum
          ) =>
            datumNumber(
              datum,
              "x"
            ) !==
              null &&
            datumNumber(
              datum,
              "y"
            ) !==
              null
        ).length >
        1
      );


    case "lorenz":
      return (
        data.filter(
          (
            datum
          ) =>
            datumNumber(
              datum,
              "population_share"
            ) !==
              null &&
            datumNumber(
              datum,
              "revenue_share"
            ) !==
              null
        ).length >
        1
      );


    case "bar":
    case "grouped_summary":
    case "heatmap":
    case "boxplot":
    case "histogram":
      return (
        data.length >
        0
      );


    case "distribution":
      return true;


    default:
      return false;
  }
}


export default function FindingCard({
  finding,
  index,
  ragContext,
}: {
  finding:
    ReportFinding;

  index:
    number;

  ragContext:
    FindingRagContext |
    null;
}) {
  return (
    <article
      className={
        styles.explanationCard
      }
    >
      <div
        className={
          styles.chartHeader
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Analyse
            {" "}
            {
              String(
                index + 1
              ).padStart(
                2,
                "0"
              )
            }
          </span>

          <h3
            className={
              styles.explanationTitle
            }
          >
            {
              finding.title
            }
          </h3>

          <p
            className={
              styles.resultSubtitle
            }
          >
            {
              familyLabel(
                finding.family
              )
            }

            {
              finding.datasets.length >
              0
                ? (
                    <>
                      {" · "}
                      {
                        finding
                          .datasets
                          .join(
                            " · "
                          )
                      }
                    </>
                  )
                : null
            }
          </p>
        </div>
      </div>


      {
        finding.summary.length >
        0
          ? (
              <div
                className={
                  styles.technicalReasons
                }
              >
                {
                  finding
                    .summary
                    .slice(
                      0,
                      3
                    )
                    .map(
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
              </div>
            )
          : null
      }


      {
        findingHasDetailedVisualization(
          finding
        )
          ? (
              <div
                className={
                  styles.chartPanel
                }
              >
                <ExpandableChart
                  title={
                    finding.title
                  }
                >
                  <FindingChart
                    finding={
                      finding
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      <RagContextBlock
        context={
          ragContext
        }
      />
    </article>
  );
}
