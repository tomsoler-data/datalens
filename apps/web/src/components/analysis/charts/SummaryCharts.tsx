import type {
  ReportChartDatum,
  ReportFinding,
} from "../../../app/types";

import styles from "../../../app/page.module.css";

import {
  datumLabel,
  datumNumber,
  formatChartNumber,
  formatDecimal,
  friendlyVariableLabel,
  metricNumber,
  metricString,
} from "../analysisPresentation";


type GroupSummaryPoint = {
  group: string;
  median: number;
};


export function GroupedSummaryChart({
  data,
}: {
  data:
    ReportChartDatum[];
}) {
  const points =
    data
      .map(
        (
          datum
        ) => {
          const group =
            datumLabel(
              datum,
              "group"
            );

          const median =
            datumNumber(
              datum,
              "median"
            );


          if (
            !group ||
            median ===
              null
          ) {
            return null;
          }


          return {
            group,
            median,
          };
        }
      )
      .filter(
        (
          point
        ): point is
          GroupSummaryPoint =>
            point !==
            null
      );


  if (
    points.length ===
    0
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Aucun groupe exploitable.
      </div>
    );
  }


  const maxValue =
    Math.max(
      ...points.map(
        (
          point
        ) =>
          point.median
      ),
      1
    );


  return (
    <div
      className={
        styles.technicalReasons
      }
    >
      {
        points.map(
          (
            point
          ) => (
            <div
              key={
                point.group
              }
              className={
                styles.summaryItem
              }
            >
              <span>
                {
                  point.group
                }
              </span>

              <strong>
                {
                  formatDecimal(
                    point.median
                  )
                }
              </strong>

              <div
                style={{
                  width:
                    "100%",

                  height:
                    "8px",

                  borderRadius:
                    "999px",

                  overflow:
                    "hidden",

                  background:
                    "rgba(255,255,255,0.08)",
                }}
              >
                <div
                  style={{
                    width:
                      `${Math.max(
                        2,
                        (
                          point.median /
                          maxValue
                        ) *
                          100
                      )}%`,

                    height:
                      "100%",

                    borderRadius:
                      "999px",

                    background:
                      "currentColor",

                    opacity:
                      0.72,
                  }}
                />
              </div>
            </div>
          )
        )
      }
    </div>
  );
}


export function GapSummaryChart({
  finding,
}: {
  finding:
    ReportFinding;
}) {
  const minimum =
    metricNumber(
      finding.metrics,
      "minimum_gap"
    );

  const median =
    metricNumber(
      finding.metrics,
      "median_gap"
    );

  const mean =
    metricNumber(
      finding.metrics,
      "mean_gap"
    );

  const maximum =
    metricNumber(
      finding.metrics,
      "maximum_gap"
    );


  if (
    minimum ===
      null ||
    median ===
      null ||
    maximum ===
      null
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Les statistiques de
        distribution sont disponibles
        dans le résumé.
      </div>
    );
  }


  const range =
    maximum -
      minimum ||
    1;


  const medianPosition =
    (
      (
        median -
        minimum
      ) /
      range
    ) *
    100;


  const meanPosition =
    mean !==
      null
      ? (
          (
            mean -
            minimum
          ) /
          range
        ) *
          100
      : null;


  return (
    <div
      className={
        styles.summaryPanel
      }
    >
      <div
        className={
          styles.summaryItem
        }
      >
        <span>
          Étendue observée
        </span>

        <strong>
          {
            formatDecimal(
              minimum
            )
          }
          {" — "}
          {
            formatDecimal(
              maximum
            )
          }
        </strong>


        <div
          style={{
            position:
              "relative",

            height:
              "18px",

            marginTop:
              "18px",

            borderRadius:
              "999px",

            background:
              "rgba(255,255,255,0.08)",
          }}
        >
          <span
            title={
              `Médiane : ${formatDecimal(
                median
              )}`
            }
            style={{
              position:
                "absolute",

              left:
                `${medianPosition}%`,

              top:
                "-5px",

              width:
                "4px",

              height:
                "28px",

              borderRadius:
                "999px",

              background:
                "currentColor",
            }}
          />


          {
            mean !==
              null &&
            meanPosition !==
              null
              ? (
                  <span
                    title={
                      `Moyenne : ${formatDecimal(
                        mean
                      )}`
                    }
                    style={{
                      position:
                        "absolute",

                      left:
                        `${meanPosition}%`,

                      top:
                        "1px",

                      width:
                        "10px",

                      height:
                        "10px",

                      borderRadius:
                        "50%",

                      border:
                        "2px solid currentColor",
                    }}
                  />
                )
              : null
          }
        </div>


        <p>
          Médiane :
          {" "}
          {
            formatDecimal(
              median
            )
          }

          {
            mean !==
              null
              ? (
                  <>
                    {" · "}
                    Moyenne :
                    {" "}
                    {
                      formatDecimal(
                        mean
                      )
                    }
                  </>
                )
              : null
          }
        </p>
      </div>
    </div>
  );
}


export function SimpleBarChart({
  finding,
}: {
  finding:
    ReportFinding;
}) {
  const groupColumn =
    metricString(
      finding.metrics,
      "group_column"
    );

  const measureColumn =
    metricString(
      finding.metrics,
      "measure_column"
    );


  const points =
    (
      finding.chart_data ??
      []
    )
      .map(
        (
          datum,
          index
        ) => {
          const label =
            datumLabel(
              datum,
              "group"
            ) ??
            (
              groupColumn
                ? datumLabel(
                    datum,
                    groupColumn
                  )
                : null
            ) ??
            datumLabel(
              datum,
              "category"
            ) ??
            datumLabel(
              datum,
              "label"
            ) ??
            `Groupe ${index + 1}`;


          const preferredKeys = [
            "value",
            measureColumn,
            "total",
            "amount",
            "sum",
            "count",
            "y",
          ].filter(
            (
              key
            ): key is string =>
              Boolean(
                key
              )
          );


          let value:
            number |
            null =
              null;


          for (
            const key
            of preferredKeys
          ) {
            value =
              datumNumber(
                datum,
                key
              );


            if (
              value !==
              null
            ) {
              break;
            }
          }


          if (
            value ===
            null
          ) {
            const excludedKeys =
              new Set(
                [
                  "group",
                  groupColumn,
                  "category",
                  "label",
                  "share",
                  "percentage",
                  "percent",
                  "n",
                ].filter(
                  (
                    key
                  ): key is string =>
                    Boolean(
                      key
                    )
                )
              );


            for (
              const [
                key,
                rawValue,
              ]
              of Object.entries(
                datum
              )
            ) {
              if (
                excludedKeys.has(
                  key
                )
              ) {
                continue;
              }


              if (
                typeof rawValue ===
                  "number" &&
                Number.isFinite(
                  rawValue
                )
              ) {
                value =
                  rawValue;

                break;
              }
            }
          }


          if (
            value ===
            null
          ) {
            return null;
          }


          return {
            label,
            value,
          };
        }
      )
      .filter(
        (
          point
        ): point is {
          label:
            string;

          value:
            number;
        } =>
          point !==
          null
      );


  if (
    points.length ===
    0
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Aucune catégorie exploitable
        pour ce graphique.
      </div>
    );
  }


  const maxValue =
    Math.max(
      ...points.map(
        (
          point
        ) =>
          Math.abs(
            point.value
          )
      ),
      1
    );


  return (
    <div
      className={
        styles.technicalReasons
      }
    >
      {
        points.map(
          (
            point,
            index
          ) => (
            <div
              key={
                `${point.label}-${index}`
              }
              className={
                styles.summaryItem
              }
              title={
                `${
                  groupColumn
                    ? `${friendlyVariableLabel(
                        groupColumn
                      )}: `
                    : ""
                }${point.label} · ${
                  measureColumn
                    ? `${friendlyVariableLabel(
                        measureColumn
                      )}: `
                    : ""
                }${formatDecimal(
                  point.value
                )}`
              }
            >
              <span>
                {
                  groupColumn
                    ? `${friendlyVariableLabel(
                        groupColumn
                      )} · ${point.label}`
                    : point.label
                }
              </span>

              <strong>
                {
                  formatChartNumber(
                    point.value
                  )
                }
              </strong>

              <div
                style={{
                  width:
                    "100%",

                  height:
                    "9px",

                  borderRadius:
                    "999px",

                  overflow:
                    "hidden",

                  background:
                    "rgba(255,255,255,0.08)",
                }}
                aria-hidden="true"
              >
                <div
                  style={{
                    width:
                      `${Math.max(
                        2,
                        (
                          Math.abs(
                            point.value
                          ) /
                          maxValue
                        ) *
                          100
                      )}%`,

                    height:
                      "100%",

                    borderRadius:
                      "999px",

                    background:
                      "currentColor",

                    opacity:
                      0.72,
                  }}
                />
              </div>
            </div>
          )
        )
      }
    </div>
  );
}
