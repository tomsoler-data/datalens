"use client";

import type { ReportBlockedAnalysis }
  from "../../app/types";

import { formatDecimal }
  from "./analysisPresentation";

import { familyLabel }
  from "./analysisPlanningPresentation";

import styles
  from "../../app/page.module.css";


export default function BlockedAnalysisList({
  items,
}: {
  items:
    ReportBlockedAnalysis[];
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
        Analyses non exécutées
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
                  {
                    familyLabel(
                      item.family
                    )
                  }
                </span>

                <h3
                  className={
                    styles.explanationTitle
                  }
                >
                  {
                    item.title
                  }
                </h3>


                <p
                  className={
                    styles.resultSubtitle
                  }
                >
                  {
                    item
                      .datasets
                      .join(
                        " · "
                      )
                  }
                </p>


                <p
                  className={
                    styles.explanationText
                  }
                >
                  {
                    item.reason
                  }
                </p>


                {
                  item
                    .caveats
                    .length >
                  0
                    ? (
                        <div
                          className={
                            styles.technicalReasons
                          }
                        >
                          {
                            item
                              .caveats
                              .map(
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
                        </div>
                      )
                    : null
                }


                <p
                  className={
                    styles.explanationText
                  }
                >
                  Priorité de découverte :
                  {" "}
                  {
                    formatDecimal(
                      item.discovery_priority_score
                    )
                  }
                </p>
              </article>
            )
          )
        }
      </div>
    </details>
  );
}
