"use client";

import type { ReportFinding } from "../../app/types";
import { familyLabel } from "./analysisPlanningPresentation";
import styles from "../../app/page.module.css";


export default function CompactFindingList({
  title,
  findings,
}: {
  title:
    string;

  findings:
    ReportFinding[];
}) {
  if (
    findings.length ===
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
        {
          title
        }
        {" · "}
        {
          findings.length
        }
      </summary>


      <div
        className={
          styles.explanationGrid
        }
      >
        {
          findings.map(
            (
              finding,
              findingIndex
            ) => (
              <article
                className={
                  styles.explanationCard
                }
                key={
                  `${
                    finding.analysis_id ??
                    `${finding.family}-${finding.title}`
                  }-${findingIndex}`
                }
              >
                <span
                  className={
                    styles.eyebrow
                  }
                >
                  {
                    familyLabel(
                      finding.family
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

                {
                  finding
                    .summary
                    .slice(
                      0,
                      2
                    )
                    .map(
                      (
                        item,
                        summaryIndex
                      ) => (
                        <p
                          className={
                            styles.explanationText
                          }
                          key={
                            `${summaryIndex}-${item}`
                          }
                        >
                          {
                            item
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
