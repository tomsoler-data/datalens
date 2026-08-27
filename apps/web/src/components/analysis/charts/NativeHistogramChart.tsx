"use client";

import {
  useState,
} from "react";

import type {
  ReportChartDatum,
} from "../../../app/types";

import styles from "../../../app/page.module.css";

import {
  datumNumber,
  formatAxisNumber,
  formatNumber,
} from "../analysisPresentation";

import {
  clampChartTooltipPosition,
  SvgChartTooltip,
} from "./ChartTooltip";


export default function NativeHistogramChart({
  data,
  valueLabel,
  lowerBound =
    null,
  upperBound =
    null,
  highlightOutliers =
    false,
}: {
  data:
    ReportChartDatum[];

  valueLabel:
    string;

  lowerBound?:
    number |
    null;

  upperBound?:
    number |
    null;

  highlightOutliers?:
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


  const bins =
    data
      .map(
        (
          datum,
          index
        ) => {
          const start =
            datumNumber(
              datum,
              "bin_start"
            );

          const end =
            datumNumber(
              datum,
              "bin_end"
            );

          const count =
            datumNumber(
              datum,
              "count"
            );


          if (
            start ===
              null ||
            end ===
              null ||
            count ===
              null
          ) {
            return null;
          }


          return {
            index,
            start,
            end,
            count,
          };
        }
      )
      .filter(
        (
          bin
        ): bin is {
          index:
            number;

          start:
            number;

          end:
            number;

          count:
            number;
        } =>
          bin !==
          null
      );


  if (
    bins.length ===
    0
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Aucun intervalle exploitable
        pour l’histogramme.
      </div>
    );
  }


  const width =
    860;

  const height =
    350;


  const padding = {
    top:
      32,

    right:
      28,

    bottom:
      68,

    left:
      78,
  };


  const plotWidth =
    width -
    padding.left -
    padding.right;

  const plotHeight =
    height -
    padding.top -
    padding.bottom;


  const maxCount =
    Math.max(
      ...bins.map(
        (
          bin
        ) =>
          bin.count
      ),
      1
    );


  const barGap =
    4;


  const barWidth =
    plotWidth /
    bins.length;


  const xMin =
    bins[
      0
    ].start;


  const xMax =
    bins[
      bins.length -
      1
    ].end;


  const xRange =
    xMax -
      xMin ||
    1;


  const projectXValue = (
    value:
      number
  ) =>
    padding.left +
    (
      (
        value -
        xMin
      ) /
      xRange
    ) *
      plotWidth;


  const lowerBoundVisible =
    lowerBound !==
      null &&
    Number.isFinite(
      lowerBound
    ) &&
    lowerBound >
      xMin &&
    lowerBound <
      xMax;


  const upperBoundVisible =
    upperBound !==
      null &&
    Number.isFinite(
      upperBound
    ) &&
    upperBound >
      xMin &&
    upperBound <
      xMax;


  const projectY = (
    count:
      number
  ) =>
    padding.top +
    plotHeight -
    (
      count /
      maxCount
    ) *
      plotHeight;


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
        bins.length -
        1
      ) /
      2
    );


  const xTicks = [
    {
      index:
        0,

      value:
        bins[
          0
        ].start,

      anchor:
        "start" as const,
    },
    {
      index:
        middleIndex,

      value:
        (
          bins[
            middleIndex
          ].start +
          bins[
            middleIndex
          ].end
        ) /
        2,

      anchor:
        "middle" as const,
    },
    {
      index:
        bins.length -
        1,

      value:
        bins[
          bins.length -
          1
        ].end,

      anchor:
        "end" as const,
    },
  ];


  const hoveredBin =
    hoveredIndex !==
      null
      ? bins[
          hoveredIndex
        ] ??
        null
      : null;


  const tooltipWidth =
    230;

  const tooltipHeight =
    84;


  const tooltipPosition =
    hoveredBin &&
    hoveredIndex !==
      null
      ? clampChartTooltipPosition(
          padding.left +
            (
              hoveredIndex +
              0.5
            ) *
              barWidth,
          projectY(
            hoveredBin.count
          ),
          tooltipWidth,
          tooltipHeight,
          width,
          height
        )
      : null;


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
          `Histogramme de ${valueLabel}`
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
              const count =
                maxCount *
                ratio;

              const y =
                projectY(
                  count
                );


              return (
                <g
                  key={
                    `hist-y-${ratio}`
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
                    style={{
                      fontSize:
                        "12px",

                      opacity:
                        0.82,
                    }}
                  >
                    {
                      formatNumber(
                        Math.round(
                          count
                        )
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


        {
          highlightOutliers &&
          lowerBoundVisible &&
          lowerBound !==
            null
            ? (
                <>
                  <rect
                    x={
                      padding.left
                    }
                    y={
                      padding.top
                    }
                    width={
                      Math.max(
                        0,
                        projectXValue(
                          lowerBound
                        ) -
                        padding.left
                      )
                    }
                    height={
                      plotHeight
                    }
                    className={
                      styles.outlierZone
                    }
                  />

                  <line
                    x1={
                      projectXValue(
                        lowerBound
                      )
                    }
                    y1={
                      padding.top
                    }
                    x2={
                      projectXValue(
                        lowerBound
                      )
                    }
                    y2={
                      padding.top +
                      plotHeight
                    }
                    className={
                      styles.outlierThreshold
                    }
                  />
                </>
              )
            : null
        }


        {
          highlightOutliers &&
          upperBoundVisible &&
          upperBound !==
            null
            ? (
                <>
                  <rect
                    x={
                      projectXValue(
                        upperBound
                      )
                    }
                    y={
                      padding.top
                    }
                    width={
                      Math.max(
                        0,
                        padding.left +
                        plotWidth -
                        projectXValue(
                          upperBound
                        )
                      )
                    }
                    height={
                      plotHeight
                    }
                    className={
                      styles.outlierZone
                    }
                  />

                  <line
                    x1={
                      projectXValue(
                        upperBound
                      )
                    }
                    y1={
                      padding.top
                    }
                    x2={
                      projectXValue(
                        upperBound
                      )
                    }
                    y2={
                      padding.top +
                      plotHeight
                    }
                    className={
                      styles.outlierThreshold
                    }
                  />
                </>
              )
            : null
        }


        {
          bins.map(
            (
              bin,
              index
            ) => {
              const x =
                padding.left +
                index *
                  barWidth +
                barGap /
                  2;

              const y =
                projectY(
                  bin.count
                );

              const heightValue =
                padding.top +
                plotHeight -
                y;


              return (
                <g
                  key={
                    `${
                      bin.index
                    }-${
                      bin.start
                    }-${
                      bin.end
                    }`
                  }
                >
                  <rect
                    x={
                      x
                    }
                    y={
                      y
                    }
                    width={
                      Math.max(
                        1,
                        barWidth -
                        barGap
                      )
                    }
                    height={
                      Math.max(
                        1,
                        heightValue
                      )
                    }
                    rx="4"
                    fill="currentColor"
                    opacity={
                      hoveredIndex ===
                        index
                        ? 0.92
                        : 0.68
                    }
                    onMouseEnter={
                      () =>
                        setHoveredIndex(
                          index
                        )
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
                    tabIndex={
                      0
                    }
                    aria-label={
                      `${formatAxisNumber(
                        bin.start
                      )} à ${formatAxisNumber(
                        bin.end
                      )}, ${formatNumber(
                        bin.count
                      )} observations`
                    }
                  >
                    <title>
                      {
                        `${
                          formatAxisNumber(
                            bin.start
                          )
                        } — ${
                          formatAxisNumber(
                            bin.end
                          )
                        } · ${
                          formatNumber(
                            bin.count
                          )
                        } observation(s)`
                      }
                    </title>
                  </rect>


                  {
                    bins.length <=
                    20
                      ? (
                          <text
                            x={
                              x +
                              Math.max(
                                1,
                                barWidth -
                                barGap
                              ) /
                              2
                            }
                            y={
                              Math.max(
                                14,
                                y -
                                7
                              )
                            }
                            textAnchor="middle"
                            className={
                              styles.chartTick
                            }
                            style={{
                              fontSize:
                                "10px",

                              opacity:
                                0.75,
                            }}
                          >
                            {
                              formatNumber(
                                bin.count
                              )
                            }
                          </text>
                        )
                      : null
                  }
                </g>
              );
            }
          )
        }


        {
          hoveredBin &&
          tooltipPosition &&
          hoveredIndex !==
            null
            ? (
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
                      `${valueLabel}`,
                      `Intervalle : ${formatAxisNumber(
                        hoveredBin.start
                      )} — ${formatAxisNumber(
                        hoveredBin.end
                      )}`,
                      `Observations : ${formatNumber(
                        hoveredBin.count
                      )}`,
                    ]
                  }
                />
              )
            : null
        }


        {
          xTicks.map(
            (
              tick
            ) => {
              const x =
                tick.index ===
                  bins.length -
                    1
                  ? padding.left +
                    plotWidth
                  : (
                      tick.index ===
                        0
                        ? padding.left
                        : padding.left +
                          (
                            tick.index +
                            0.5
                          ) *
                            barWidth
                    );


              return (
                <text
                  key={
                    `hist-x-${tick.index}`
                  }
                  x={
                    x
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
                    formatAxisNumber(
                      tick.value
                    )
                  }
                </text>
              );
            }
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
            10
          }
          textAnchor="middle"
          className={
            styles.chartTick
          }
          style={{
            fontSize:
              "12px",

            opacity:
              0.76,
          }}
        >
          {
            valueLabel
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
          style={{
            fontSize:
              "12px",

            opacity:
              0.76,
          }}
        >
          Observations
        </text>
      </svg>
    </div>
  );
}
