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
  formatTemporalDisplayValue,
} from "../analysisPresentation";

import {
  clampChartTooltipPosition,
  SvgChartTooltip,
} from "./ChartTooltip";


export default function RequestedTimeSeriesChart({
  data,
  valueLabel,
  showMovingAverage =
    false,
}: {
  data:
    ReportChartDatum[];

  valueLabel:
    string;

  showMovingAverage?:
    boolean;
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


  const points =
    data
      .map(
        (
          datum,
          index
        ) => {
          const label =
            datumLabel(
              datum,
              "period"
            ) ??
            String(
              index + 1
            );

          const value =
            datumNumber(
              datum,
              "value"
            );

          const movingAverage =
            datumNumber(
              datum,
              "moving_average"
            );


          if (
            value ===
            null
          ) {
            return null;
          }


          return {
            label,
            value,
            movingAverage,
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

          movingAverage:
            number |
            null;
        } =>
          point !==
          null
      );


  if (
    points.length <
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


  const values = [
    ...points.map(
      (
        point
      ) =>
        point.value
    ),

    ...points
      .map(
        (
          point
        ) =>
          point.movingAverage
      )
      .filter(
        (
          value
        ): value is number =>
          value !==
          null
      ),
  ];


  const width =
    860;

  const height =
    330;


  const padding = {
    top:
      28,

    right:
      30,

    bottom:
      58,

    left:
      82,
  };


  const yMin =
    Math.min(
      ...values
    );

  const yMax =
    Math.max(
      ...values
    );

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
      index /
      Math.max(
        points.length -
          1,
        1
      )
    ) *
      plotWidth;


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


  const primaryLine =
    points
      .map(
        (
          point,
          index
        ) =>
          `${projectX(
            index
          )},${projectY(
            point.value
          )}`
      )
      .join(
        " "
      );


  const movingLine =
    points
      .map(
        (
          point,
          index
        ) => {
          if (
            point.movingAverage ===
            null
          ) {
            return null;
          }


          return `${projectX(
            index
          )},${projectY(
            point.movingAverage
          )}`;
        }
      )
      .filter(
        (
          point
        ): point is string =>
          point !==
          null
      )
      .join(
        " "
      );


  const tickRatios = [
    0,
    0.25,
    0.5,
    0.75,
    1,
  ];


  const xTickIndexes =
    Array.from(
      new Set(
        [
          0,
          Math.floor(
            (
              points.length -
              1
            ) /
            2
          ),
          points.length -
            1,
        ]
      )
    );


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
    hoveredPoint?.movingAverage !==
      null &&
    hoveredPoint?.movingAverage !==
      undefined
      ? 3
      : 2;

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
            hoveredPoint.value
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
          `Évolution de ${valueLabel}`
        }
        onMouseLeave={
          () =>
            setHoveredIndex(
              null
            )
        }
      >
        {
          tickRatios.map(
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
                    ratio
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
                      10
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


        <polyline
          points={
            primaryLine
          }
          fill="none"
          stroke="currentColor"
          strokeWidth="2.4"
          strokeLinecap="round"
          strokeLinejoin="round"
          opacity={
            showMovingAverage
              ? 0.48
              : 0.92
          }
        />


        {
          showMovingAverage &&
          movingLine
            ? (
                <>
                  <polyline
                    points={
                      movingLine
                    }
                    fill="none"
                    stroke="#e8b861"
                    strokeWidth="8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    opacity="0.13"
                    pointerEvents="none"
                  />

                  <polyline
                    points={
                      movingLine
                    }
                    fill="none"
                    stroke="#e8b861"
                    strokeWidth="4"
                    strokeDasharray="12 7"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    opacity="1"
                    pointerEvents="none"
                    style={{
                      filter:
                        "drop-shadow(0 0 3px rgba(232, 184, 97, 0.34))",
                    }}
                  />
                </>
              )
            : null
        }


        {
          points.map(
            (
              point,
              index
            ) => (
              <g
                key={
                  `${point.label}-${index}`
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
                      point.value
                    )
                  }
                  r={
                    hoveredIndex ===
                      index
                      ? 5
                      : 3.2
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
                    `${formatTemporalDisplayValue(
                      point.label
                    )}, ${valueLabel}: ${formatDecimal(
                      point.value
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
                          hoveredPoint.label
                        ),
                        `${valueLabel} : ${formatDecimal(
                          hoveredPoint.value
                        )}`,
                        ...(
                          hoveredPoint.movingAverage !==
                            null
                            ? [
                                `Moyenne mobile : ${formatDecimal(
                                  hoveredPoint.movingAverage
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
          xTickIndexes.map(
            (
              index
            ) => (
              <text
                key={
                  index
                }
                x={
                  projectX(
                    index
                  )
                }
                y={
                  padding.top +
                  plotHeight +
                  22
                }
                textAnchor={
                  index ===
                    0
                    ? "start"
                    : (
                        index ===
                        points.length -
                          1
                          ? "end"
                          : "middle"
                      )
                }
                className={
                  styles.chartTick
                }
              >
                {
                  formatTemporalDisplayValue(
                    points[
                      index
                    ].label
                  )
                }
              </text>
            )
          )
        }


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
          Période
        </text>


        {
          showMovingAverage
            ? (
                <g
                  transform={
                    `translate(${padding.left + 8} 12)`
                  }
                >
                  <line
                    x1="0"
                    y1="0"
                    x2="28"
                    y2="0"
                    stroke="currentColor"
                    strokeWidth="3"
                  />

                  <text
                    x="36"
                    y="4"
                    className={
                      styles.chartTick
                    }
                  >
                    Valeur
                  </text>

                  <line
                    x1="105"
                    y1="0"
                    x2="133"
                    y2="0"
                    stroke="#e8b861"
                    strokeWidth="4"
                    strokeDasharray="10 6"
                    opacity="1"
                    style={{
                      filter:
                        "drop-shadow(0 0 3px rgba(232, 184, 97, 0.34))",
                    }}
                  />

                  <text
                    x="141"
                    y="4"
                    className={
                      styles.chartTick
                    }
                    style={{
                      fill:
                        "#e8b861",

                      fontWeight:
                        700,
                    }}
                  >
                    Moyenne mobile
                  </text>
                </g>
              )
            : null
        }
      </svg>
    </div>
  );
}
