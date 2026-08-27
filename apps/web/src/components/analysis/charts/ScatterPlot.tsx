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
  formatDecimal,
} from "../analysisPresentation";

import {
  clampChartTooltipPosition,
  SvgChartTooltip,
} from "./ChartTooltip";


type ChartPoint = {
  x: number;
  y: number;
};


export default function ScatterPlot({
  data,
  xLabel =
    "Variable X",
  yLabel =
    "Variable Y",
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
          datum
        ) => {
          const x =
            datumNumber(
              datum,
              "x"
            );

          const y =
            datumNumber(
              datum,
              "y"
            );


          if (
            x ===
              null ||
            y ===
              null
          ) {
            return null;
          }


          return {
            x,
            y,
          };
        }
      )
      .filter(
        (
          point
        ): point is
          ChartPoint =>
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
        Pas assez de points
        pour afficher cette relation.
      </div>
    );
  }


  const width =
    860;

  const height =
    300;


  const padding = {
    top:
      18,

    right:
      24,

    bottom:
      60,

    left:
      92,
  };


  const xMin =
    Math.min(
      ...points.map(
        (
          point
        ) =>
          point.x
      )
    );

  const xMax =
    Math.max(
      ...points.map(
        (
          point
        ) =>
          point.x
      )
    );

  const yMin =
    Math.min(
      ...points.map(
        (
          point
        ) =>
          point.y
      )
    );

  const yMax =
    Math.max(
      ...points.map(
        (
          point
        ) =>
          point.y
      )
    );


  const xRange =
    xMax -
      xMin ||
    1;

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


  const tickRatios = [
    0,
    0.25,
    0.5,
    0.75,
    1,
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
    220;

  const tooltipHeight =
    66;


  const tooltipPosition =
    hoveredPoint
      ? clampChartTooltipPosition(
          projectX(
            hoveredPoint.x
          ),
          projectY(
            hoveredPoint.y
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
          `Nuage de points : ${xLabel} et ${yLabel}`
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
              const x =
                padding.left +
                ratio *
                  plotWidth;

              const y =
                padding.top +
                plotHeight -
                ratio *
                  plotHeight;

              const xValue =
                xMin +
                ratio *
                  xRange;

              const yValue =
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
                      padding.top +
                      plotHeight +
                      20
                    }
                    textAnchor="middle"
                    className={
                      styles.chartTick
                    }
                  >
                    {
                      formatAxisNumber(
                        xValue
                      )
                    }
                  </text>

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
                        yValue
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
          points
            .slice(
              0,
              4000
            )
            .map(
              (
                point,
                index
              ) => (
                <g
                  key={
                    `${point.x}-${point.y}-${index}`
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
                    `${xLabel}: ${formatDecimal(
                      point.x
                    )}, ${yLabel}: ${formatDecimal(
                      point.y
                    )}`
                  }
                  style={{
                    outline:
                      "none",
                  }}
                >
                  <circle
                    cx={
                      projectX(
                        point.x
                      )
                    }
                    cy={
                      projectY(
                        point.y
                      )
                    }
                    r={
                      hoveredIndex ===
                        index
                        ? 5
                        : 3.1
                    }
                    className={
                      styles.chartPoint
                    }
                  />

                  <circle
                    cx={
                      projectX(
                        point.x
                      )
                    }
                    cy={
                      projectY(
                        point.y
                      )
                    }
                    r="10"
                    fill="transparent"
                  />
                </g>
              )
            )
        }


        {
          hoveredPoint &&
          tooltipPosition
            ? (
                <>
                  <line
                    x1={
                      projectX(
                        hoveredPoint.x
                      )
                    }
                    y1={
                      padding.top
                    }
                    x2={
                      projectX(
                        hoveredPoint.x
                      )
                    }
                    y2={
                      padding.top +
                      plotHeight
                    }
                    stroke="currentColor"
                    strokeWidth="1"
                    opacity="0.18"
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
                        "Observation",
                        `${xLabel} : ${formatDecimal(
                          hoveredPoint.x
                        )}`,
                        `${yLabel} : ${formatDecimal(
                          hoveredPoint.y
                        )}`,
                      ]
                    }
                  />
                </>
              )
            : null
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
          x="18"
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
            `rotate(-90 18 ${
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
