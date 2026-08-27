"use client";

import {
  useState,
} from "react";

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
  formatTemporalDisplayValue,
} from "../analysisPresentation";

import {
  clampChartTooltipPosition,
  SvgChartTooltip,
} from "./ChartTooltip";


export function lineBandRenderablePoints(
  data:
    ReportChartDatum[]
) {
  return (
    data
      .map(
        (
          datum
        ) => {
          const period =
            datumLabel(
              datum,
              "period"
            );

          const median =
            datumNumber(
              datum,
              "median"
            );

          const q1 =
            datumNumber(
              datum,
              "q1"
            );

          const q3 =
            datumNumber(
              datum,
              "q3"
            );

          const count =
            datumNumber(
              datum,
              "count"
            );


          if (
            period ===
              null ||
            median ===
              null ||
            q1 ===
              null ||
            q3 ===
              null
          ) {
            return null;
          }


          return {
            period,
            median,
            q1,
            q3,
            count,
          };
        }
      )
      .filter(
        (
          point
        ): point is {
          period:
            string;

          median:
            number;

          q1:
            number;

          q3:
            number;

          count:
            number |
            null;
        } =>
          point !==
          null
      )
  );
}


function downsampleLineBandPoints(
  points:
    ReturnType<
      typeof lineBandRenderablePoints
    >,

  maxPoints =
    160
) {
  if (
    points.length <=
    maxPoints
  ) {
    return points;
  }


  const step =
    (
      points.length -
      1
    ) /
    (
      maxPoints -
      1
    );


  return (
    Array.from(
      {
        length:
          maxPoints,
      },
      (
        _,
        index
      ) =>
        points[
          Math.round(
            index *
            step
          )
        ]
    )
  );
}


export default function LineBandChart({
  data,
  yLabel =
    "Valeur",
}: {
  data:
    ReportChartDatum[];

  yLabel?:
    string;
}) {
  const [
    hoveredIndex,
    setHoveredIndex,
  ] = useState<
    number |
    null
  >(
    null
  );


  const allPoints =
    lineBandRenderablePoints(
      data
    );


  if (
    allPoints.length <
    2
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Pas assez de périodes
        pour afficher l’évolution.
      </div>
    );
  }


  const points =
    downsampleLineBandPoints(
      allPoints
    );


  const width =
    860;

  const height =
    420;


  const padding = {
    top:
      28,

    right:
      28,

    bottom:
      58,

    left:
      94,
  };


  const values =
    points.flatMap(
      (
        point
      ) => [
        point.q1,
        point.median,
        point.q3,
      ]
    );


  const rawMin =
    Math.min(
      ...values
    );

  const rawMax =
    Math.max(
      ...values
    );


  const rawRange =
    rawMax -
      rawMin ||
    1;


  const yPadding =
    rawRange *
    0.06;


  const yMin =
    rawMin -
    yPadding;

  const yMax =
    rawMax +
    yPadding;


  const yRange =
    yMax -
      yMin ||
    1;


  const plotWidth =
    width -
    padding.left -
    padding.right;

  const plotHeight =
    height -
    padding.top -
    padding.bottom;


  const projectX = (
    index:
      number
  ) =>
    padding.left +
    (
      points.length ===
      1
        ? 0
        : (
            index /
            (
              points.length -
              1
            )
          ) *
          plotWidth
    );


  const projectY = (
    value:
      number
  ) =>
    padding.top +
    plotHeight -
    (
      (
        value -
        yMin
      ) /
      yRange
    ) *
      plotHeight;


  const upper =
    points.map(
      (
        point,
        index
      ) =>
        `${projectX(
          index
        )},${projectY(
          point.q3
        )}`
    );


  const lower =
    [...points]
      .reverse()
      .map(
        (
          point,
          reverseIndex
        ) => {
          const index =
            points.length -
            1 -
            reverseIndex;


          return (
            `${projectX(
              index
            )},${projectY(
              point.q1
            )}`
          );
        }
      );


  const band =
    [
      ...upper,
      ...lower,
    ].join(
      " "
    );


  const medianLine =
    points
      .map(
        (
          point,
          index
        ) =>
          `${projectX(
            index
          )},${projectY(
            point.median
          )}`
      )
      .join(
        " "
      );


  const yTickRatios = [
    0,
    0.25,
    0.5,
    0.75,
    1,
  ];


  const middleIndex =
    Math.floor(
      (
        points.length -
        1
      ) /
      2
    );


  const xTicks = [
    {
      index:
        0,

      anchor:
        "start" as const,
    },
    {
      index:
        middleIndex,

      anchor:
        "middle" as const,
    },
    {
      index:
        points.length -
        1,

      anchor:
        "end" as const,
    },
  ];


  const hoveredPoint =
    hoveredIndex !==
      null
      ? points[
          hoveredIndex
        ] ??
        null
      : null;


  const tooltipWidth =
    230;

  const tooltipLineCount =
    hoveredPoint?.count !==
      null &&
    hoveredPoint?.count !==
      undefined
      ? 5
      : 4;

  const tooltipHeight =
    24 +
    tooltipLineCount *
      18;


  const tooltipPosition =
    hoveredPoint &&
    hoveredIndex !==
      null
      ? clampChartTooltipPosition(
          projectX(
            hoveredIndex
          ),
          projectY(
            hoveredPoint.median
          ),
          tooltipWidth,
          tooltipHeight,
          width,
          height
        )
      : null;


  const hoverBandWidth =
    plotWidth /
    Math.max(
      points.length -
        1,
      1
    );


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
          `Évolution temporelle de ${yLabel} sur ${
            formatNumber(
              allPoints.length
            )
          } période(s)`
        }
        onMouseLeave={
          () =>
            setHoveredIndex(
              null
            )
        }
      >
        {
          yTickRatios.map(
            (
              ratio
            ) => {
              const y =
                padding.top +
                plotHeight -
                ratio *
                  plotHeight;

              const value =
                yMin +
                ratio *
                  yRange;


              return (
                <g
                  key={
                    `y-${ratio}`
                  }
                >
                  <line
                    x1={
                      padding.left
                    }
                    y1={
                      y
                    }
                    x2={
                      padding.left +
                      plotWidth
                    }
                    y2={
                      y
                    }
                    className={
                      styles.chartGrid
                    }
                  />

                  <text
                    x={
                      padding.left -
                      12
                    }
                    y={
                      y +
                      4
                    }
                    textAnchor="end"
                    className={
                      styles.chartTick
                    }
                    style={{
                      fontSize:
                        "12px",

                      opacity:
                        0.82,
                    }}
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


        <line
          x1={
            padding.left
          }
          y1={
            padding.top
          }
          x2={
            padding.left
          }
          y2={
            padding.top +
            plotHeight
          }
          className={
            styles.chartAxis
          }
        />


        <polygon
          points={
            band
          }
          fill="currentColor"
          opacity="0.08"
        />


        <polyline
          points={
            medianLine
          }
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinejoin="round"
          strokeLinecap="round"
        />


        {
          points.map(
            (
              point,
              index
            ) => (
              <g
                key={
                  `${
                    point.period
                  }-${
                    index
                  }`
                }
              >
                <rect
                  x={
                    Math.max(
                      padding.left,
                      projectX(
                        index
                      ) -
                      hoverBandWidth /
                        2
                    )
                  }
                  y={
                    padding.top
                  }
                  width={
                    Math.min(
                      hoverBandWidth,
                      padding.left +
                        plotWidth -
                        Math.max(
                          padding.left,
                          projectX(
                            index
                          ) -
                          hoverBandWidth /
                            2
                        )
                    )
                  }
                  height={
                    plotHeight
                  }
                  fill="transparent"
                  onMouseEnter={
                    () =>
                      setHoveredIndex(
                        index
                      )
                  }
                />

                <circle
                  cx={
                    projectX(
                      index
                    )
                  }
                  cy={
                    projectY(
                      point.median
                    )
                  }
                  r={
                    hoveredIndex ===
                      index
                      ? 5
                      : 3.4
                  }
                  className={
                    styles.chartPoint
                  }
                  tabIndex={
                    0
                  }
                  onFocus={
                    () =>
                      setHoveredIndex(
                        index
                      )
                  }
                  onBlur={
                    () =>
                      setHoveredIndex(
                        null
                      )
                  }
                  aria-label={
                    `${
                      formatTemporalDisplayValue(
                        point.period
                      )
                    }, médiane ${formatDecimal(
                      point.median
                    )}, Q1 ${formatDecimal(
                      point.q1
                    )}, Q3 ${formatDecimal(
                      point.q3
                    )}`
                  }
                />
              </g>
            )
          )
        }


        {
          hoveredPoint &&
          tooltipPosition &&
          hoveredIndex !==
            null
            ? (
                <>
                  <line
                    x1={
                      projectX(
                        hoveredIndex
                      )
                    }
                    y1={
                      padding.top
                    }
                    x2={
                      projectX(
                        hoveredIndex
                      )
                    }
                    y2={
                      padding.top +
                      plotHeight
                    }
                    stroke="currentColor"
                    strokeWidth="1"
                    opacity="0.24"
                    pointerEvents="none"
                  />

                  <circle
                    cx={
                      projectX(
                        hoveredIndex
                      )
                    }
                    cy={
                      projectY(
                        hoveredPoint.median
                      )
                    }
                    r="7"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    opacity="0.7"
                    pointerEvents="none"
                  />

                  <SvgChartTooltip
                    x={
                      tooltipPosition.x
                    }
                    y={
                      tooltipPosition.y
                    }
                    width={
                      tooltipWidth
                    }
                    lines={
                      [
                        formatTemporalDisplayValue(
                          hoveredPoint.period
                        ),
                        `Médiane : ${formatDecimal(
                          hoveredPoint.median
                        )}`,
                        `Q1 : ${formatDecimal(
                          hoveredPoint.q1
                        )}`,
                        `Q3 : ${formatDecimal(
                          hoveredPoint.q3
                        )}`,
                        ...(
                          hoveredPoint.count !==
                            null
                            ? [
                                `Observations : ${formatNumber(
                                  hoveredPoint.count
                                )}`,
                              ]
                            : []
                        ),
                      ]
                    }
                  />
                </>
              )
            : null
        }


        {
          xTicks.map(
            (
              tick
            ) => (
              <text
                key={
                  `${
                    tick.index
                  }-${
                    tick.anchor
                  }`
                }
                x={
                  projectX(
                    tick.index
                  )
                }
                y={
                  padding.top +
                  plotHeight +
                  24
                }
                textAnchor={
                  tick.anchor
                }
                className={
                  styles.chartTick
                }
                style={{
                  fontSize:
                    "12px",

                  opacity:
                    0.82,
                }}
              >
                {
                  formatTemporalDisplayValue(
                    points[
                      tick.index
                    ].period
                  )
                }
              </text>
            )
          )
        }


        <text
          x="24"
          y={
            padding.top +
            plotHeight /
              2
          }
          textAnchor="middle"
          className={
            styles.chartTick
          }
          transform={
            `rotate(-90 24 ${
              padding.top +
              plotHeight /
                2
            })`
          }
          style={{
            fontSize:
              "12px",

            opacity:
              0.76,
          }}
        >
          {
            yLabel
          }
        </text>
      </svg>


      {
        allPoints.length >
        points.length
          ? (
              <p
                style={{
                  margin:
                    "8px 0 0",

                  fontSize:
                    "0.69rem",

                  opacity:
                    0.52,
                }}
              >
                {
                  formatNumber(
                    allPoints.length
                  )
                }
                {" périodes · affichage simplifié à "}
                {
                  formatNumber(
                    points.length
                  )
                }
                {" points pour la lisibilité."}
              </p>
            )
          : null
      }
    </div>
  );
}
