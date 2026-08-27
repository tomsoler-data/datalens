import type {
  ReportChartDatum,
} from "../../../app/types";

import styles from "../../../app/page.module.css";

import {
  datumLabel,
  datumNumber,
  formatAxisNumber,
  formatDecimal,
  formatNumber,
} from "../analysisPresentation";


export default function RequestedBoxPlotChart({
  data,
  groupLabel,
  valueLabel,
}: {
  data:
    ReportChartDatum[];

  groupLabel:
    string;

  valueLabel:
    string;
}) {
  const groups =
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

          const minimum =
            datumNumber(
              datum,
              "min"
            );

          const q1 =
            datumNumber(
              datum,
              "q1"
            );

          const median =
            datumNumber(
              datum,
              "median"
            );

          const q3 =
            datumNumber(
              datum,
              "q3"
            );

          const maximum =
            datumNumber(
              datum,
              "max"
            );

          const count =
            datumNumber(
              datum,
              "count"
            );


          if (
            group ===
              null ||
            minimum ===
              null ||
            q1 ===
              null ||
            median ===
              null ||
            q3 ===
              null ||
            maximum ===
              null
          ) {
            return null;
          }


          return {
            group,
            minimum,
            q1,
            median,
            q3,
            maximum,
            count,
          };
        }
      )
      .filter(
        (
          group
        ): group is {
          group:
            string;

          minimum:
            number;

          q1:
            number;

          median:
            number;

          q3:
            number;

          maximum:
            number;

          count:
            number |
            null;
        } =>
          group !==
          null
      );


  if (
    groups.length ===
    0
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Aucun résumé de distribution
        exploitable.
      </div>
    );
  }


  const minValue =
    Math.min(
      ...groups.map(
        (
          group
        ) =>
          group.minimum
      )
    );

  const maxValue =
    Math.max(
      ...groups.map(
        (
          group
        ) =>
          group.maximum
      )
    );

  const valueRange =
    maxValue -
      minValue ||
    1;


  const width =
    860;

  const rowHeight =
    72;

  const height =
    Math.max(
      250,
      112 +
      groups.length *
        rowHeight
    );


  const padding = {
    top:
      28,

    right:
      30,

    bottom:
      62,

    left:
      130,
  };


  const plotWidth =
    width -
    padding.left -
    padding.right;

  const plotHeight =
    height -
    padding.top -
    padding.bottom;


  const projectX = (
    value:
      number
  ) =>
    padding.left +
    (
      (
        value -
        minValue
      ) /
      valueRange
    ) *
      plotWidth;


  const tickRatios = [
    0,
    0.25,
    0.5,
    0.75,
    1,
  ];


  return (
    <div
      className={
        styles.chartCanvas
      }
    >
      <svg
        viewBox={
          `0 0 ${width} ${height}`
        }
        role="img"
        aria-label={
          `Distribution de ${valueLabel} selon ${groupLabel}`
        }
      >
        {
          tickRatios.map(
            (
              ratio
            ) => {
              const x =
                padding.left +
                ratio *
                  plotWidth;

              const value =
                minValue +
                ratio *
                  valueRange;


              return (
                <g
                  key={
                    ratio
                  }
                >
                  <line
                    x1={
                      x
                    }
                    y1={
                      padding.top
                    }
                    x2={
                      x
                    }
                    y2={
                      padding.top +
                      plotHeight
                    }
                    className={
                      styles.chartGrid
                    }
                  />

                  <text
                    x={
                      x
                    }
                    y={
                      height -
                      28
                    }
                    textAnchor="middle"
                    className={
                      styles.chartTick
                    }
                  >
                    {
                      formatAxisNumber(
                        value
                      )
                    }
                  </text>
                </g>
              );
            }
          )
        }


        {
          groups.map(
            (
              group,
              index
            ) => {
              const y =
                padding.top +
                36 +
                index *
                  rowHeight;


              return (
                <g
                  key={
                    group.group
                  }
                >
                  <text
                    x={
                      padding.left -
                      14
                    }
                    y={
                      y +
                      4
                    }
                    textAnchor="end"
                    className={
                      styles.chartTick
                    }
                  >
                    {
                      group.group
                    }
                  </text>

                  <line
                    x1={
                      projectX(
                        group.minimum
                      )
                    }
                    y1={
                      y
                    }
                    x2={
                      projectX(
                        group.maximum
                      )
                    }
                    y2={
                      y
                    }
                    stroke="currentColor"
                    strokeWidth="2"
                    opacity="0.55"
                  />

                  <line
                    x1={
                      projectX(
                        group.minimum
                      )
                    }
                    y1={
                      y -
                      9
                    }
                    x2={
                      projectX(
                        group.minimum
                      )
                    }
                    y2={
                      y +
                      9
                    }
                    stroke="currentColor"
                    strokeWidth="2"
                    opacity="0.65"
                  />

                  <line
                    x1={
                      projectX(
                        group.maximum
                      )
                    }
                    y1={
                      y -
                      9
                    }
                    x2={
                      projectX(
                        group.maximum
                      )
                    }
                    y2={
                      y +
                      9
                    }
                    stroke="currentColor"
                    strokeWidth="2"
                    opacity="0.65"
                  />

                  <rect
                    x={
                      projectX(
                        group.q1
                      )
                    }
                    y={
                      y -
                      14
                    }
                    width={
                      Math.max(
                        2,
                        projectX(
                          group.q3
                        ) -
                        projectX(
                          group.q1
                        )
                      )
                    }
                    height="28"
                    rx="6"
                    fill="currentColor"
                    opacity="0.22"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <title>
                      {
                        `${groupLabel}: ${group.group} · ` +
                        `min ${formatDecimal(
                          group.minimum
                        )} · Q1 ${formatDecimal(
                          group.q1
                        )} · médiane ${formatDecimal(
                          group.median
                        )} · Q3 ${formatDecimal(
                          group.q3
                        )} · max ${formatDecimal(
                          group.maximum
                        )}` +
                        (
                          group.count !==
                            null
                            ? ` · n=${formatNumber(
                                group.count
                              )}`
                            : ""
                        )
                      }
                    </title>
                  </rect>

                  <line
                    x1={
                      projectX(
                        group.median
                      )
                    }
                    y1={
                      y -
                      17
                    }
                    x2={
                      projectX(
                        group.median
                      )
                    }
                    y2={
                      y +
                      17
                    }
                    stroke="currentColor"
                    strokeWidth="3"
                  />
                </g>
              );
            }
          )
        }


        <line
          x1={
            padding.left
          }
          y1={
            padding.top +
            plotHeight
          }
          x2={
            padding.left +
            plotWidth
          }
          y2={
            padding.top +
            plotHeight
          }
          className={
            styles.chartAxis
          }
        />


        <text
          x={
            padding.left +
            plotWidth /
              2
          }
          y={
            height -
            8
          }
          textAnchor="middle"
          className={
            styles.chartTick
          }
        >
          {
            valueLabel
          }
        </text>
      </svg>
    </div>
  );
}
