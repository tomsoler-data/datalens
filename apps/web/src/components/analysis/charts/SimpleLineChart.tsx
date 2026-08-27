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


export default function SimpleLineChart({
  data,
  xLabel =
    "Période",
  yLabel =
    "Valeur",
}: {
  data:
    ReportChartDatum[];

  xLabel?:
    string;

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


  const width =
    860;

  const height =
    330;


  const padding = {
    top:
      20,

    right:
      28,

    bottom:
      58,

    left:
      82,
  };


  const values =
    points.map(
      (
        point
      ) =>
        point.value
    );


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


  const line =
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
    220;

  const tooltipHeight =
    66;


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
          `Évolution de ${yLabel} selon ${xLabel}`
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
            line
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
                    `${formatTemporalDisplayValue(
                      point.label
                    )}, ${yLabel}: ${formatDecimal(
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
                        `${yLabel} : ${formatDecimal(
                          hoveredPoint.value
                        )}`,
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
          {
            xLabel
          }
        </text>


        <text
          x="20"
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
            `rotate(-90 20 ${
              padding.top +
              plotHeight /
                2
            })`
          }
        >
          {
            yLabel
          }
        </text>
      </svg>
    </div>
  );
}
