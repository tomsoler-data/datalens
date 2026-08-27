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
  formatPercent,
} from "../analysisPresentation";


export default function RequestedLorenzChart({
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
          const populationShare =
            datumNumber(
              datum,
              "population_share"
            );

          const revenueShare =
            datumNumber(
              datum,
              "revenue_share"
            );

          const equalityShare =
            datumNumber(
              datum,
              "equality_share"
            );


          if (
            populationShare ===
              null ||
            revenueShare ===
              null
          ) {
            return null;
          }


          return {
            populationShare,
            revenueShare,
            equalityShare:
              equalityShare ??
              populationShare,
          };
        }
      )
      .filter(
        (
          point
        ): point is {
          populationShare:
            number;

          revenueShare:
            number;

          equalityShare:
            number;
        } =>
          point !==
          null
      );


  const [
    hoveredIndex,
    setHoveredIndex,
  ] = useState<
    number |
    null
  >(
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
        pour afficher la courbe de Lorenz.
      </div>
    );
  }


  const width =
    860;

  const height =
    360;


  const padding = {
    top:
      26,

    right:
      30,

    bottom:
      62,

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


  const projectX = (
    share:
      number
  ) =>
    padding.left +
    Math.max(
      0,
      Math.min(
        1,
        share
      )
    ) *
      plotWidth;


  const projectY = (
    share:
      number
  ) =>
    padding.top +
    plotHeight -
    Math.max(
      0,
      Math.min(
        1,
        share
      )
    ) *
      plotHeight;


  const lorenzLine =
    points
      .map(
        (
          point
        ) =>
          `${projectX(
            point.populationShare
          )},${projectY(
            point.revenueShare
          )}`
      )
      .join(
        " "
      );


  const equalityLine =
    points
      .map(
        (
          point
        ) =>
          `${projectX(
            point.populationShare
          )},${projectY(
            point.equalityShare
          )}`
      )
      .join(
        " "
      );


  const ticks = [
    0,
    0.25,
    0.5,
    0.75,
    1,
  ];


  const hoveredPoint =
    hoveredIndex ===
      null
      ? null
      : points[
          hoveredIndex
        ];


  const handleMouseMove = (
    event:
      React.MouseEvent<
        SVGSVGElement
      >
  ) => {
    const rect =
      event.currentTarget
        .getBoundingClientRect();


    if (
      rect.width <=
      0
    ) {
      return;
    }


    const svgX =
      (
        (
          event.clientX -
          rect.left
        ) /
        rect.width
      ) *
      width;


    const targetShare =
      Math.max(
        0,
        Math.min(
          1,
          (
            svgX -
            padding.left
          ) /
            plotWidth
        )
      );


    let nearestIndex =
      0;

    let nearestDistance =
      Number.POSITIVE_INFINITY;


    points.forEach(
      (
        point,
        index
      ) => {
        const distance =
          Math.abs(
            point.populationShare -
            targetShare
          );


        if (
          distance <
          nearestDistance
        ) {
          nearestDistance =
            distance;

          nearestIndex =
            index;
        }
      }
    );


    setHoveredIndex(
      nearestIndex
    );
  };


  const tooltipWidth =
    205;

  const tooltipHeight =
    72;


  const hoverX =
    hoveredPoint
      ? projectX(
          hoveredPoint
            .populationShare
        )
      : 0;

  const hoverY =
    hoveredPoint
      ? projectY(
          hoveredPoint
            .revenueShare
        )
      : 0;


  const tooltipX =
    hoveredPoint
      ? (
          hoverX >
          width * 0.62
            ? Math.max(
                8,
                hoverX -
                  tooltipWidth -
                  18
              )
            : Math.min(
                width -
                  tooltipWidth -
                  8,
                hoverX +
                  18
              )
        )
      : 0;


  const tooltipY =
    hoveredPoint
      ? (
          hoverY >
          tooltipHeight +
            28
            ? hoverY -
              tooltipHeight -
              18
            : Math.min(
                height -
                  tooltipHeight -
                  8,
                hoverY +
                  18
              )
        )
      : 0;


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
          "Courbe de Lorenz du chiffre d\u2019affaires client"
        }
        onMouseMove={
          handleMouseMove
        }
        onMouseLeave={
          () =>
            setHoveredIndex(
              null
            )
        }
      >
        {
          ticks.map(
            (
              share
            ) => {
              const x =
                projectX(
                  share
                );

              const y =
                projectY(
                  share
                );


              return (
                <g
                  key={
                    share
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
                      22
                    }
                    textAnchor="middle"
                    className={
                      styles.chartTick
                    }
                  >
                    {
                      formatPercent(
                        share
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
                      formatPercent(
                        share
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
            equalityLine
          }
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeDasharray="8 8"
          opacity="0.42"
          pointerEvents="none"
        />


        <polyline
          points={
            lorenzLine
          }
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
          pointerEvents="none"
        />


        {
          points
            .filter(
              (
                _,
                index
              ) =>
                (
                  index ===
                    0 ||
                  index ===
                    points.length -
                      1 ||
                  index %
                    Math.max(
                      1,
                      Math.floor(
                        points.length /
                        24
                      )
                    )
                    ===
                    0
                )
            )
            .map(
              (
                point,
                index
              ) => (
                <circle
                  key={
                    `${point.populationShare}-${point.revenueShare}-${index}`
                  }
                  cx={
                    projectX(
                      point.populationShare
                    )
                  }
                  cy={
                    projectY(
                      point.revenueShare
                    )
                  }
                  r="3"
                  className={
                    styles.chartPoint
                  }
                  pointerEvents="none"
                />
              )
            )
        }


        {
          hoveredPoint
            ? (
                <>
                  <line
                    x1={
                      hoverX
                    }
                    y1={
                      padding.top
                    }
                    x2={
                      hoverX
                    }
                    y2={
                      padding.top +
                      plotHeight
                    }
                    stroke="currentColor"
                    strokeWidth="1"
                    strokeDasharray="4 4"
                    opacity="0.5"
                    pointerEvents="none"
                  />


                  <circle
                    cx={
                      hoverX
                    }
                    cy={
                      hoverY
                    }
                    r="6"
                    fill="currentColor"
                    stroke="white"
                    strokeWidth="2"
                    pointerEvents="none"
                  />


                  <g
                    transform={
                      `translate(${tooltipX} ${tooltipY})`
                    }
                    pointerEvents="none"
                  >
                    <rect
                      width={
                        tooltipWidth
                      }
                      height={
                        tooltipHeight
                      }
                      rx="9"
                      fill="#0B1220"
                      stroke="#334155"
                      strokeWidth="1"
                      opacity="0.98"
                    />


                    <text
                      x="12"
                      y="20"
                      fontSize="11"
                      fontWeight="700"
                      fill="#F8FAFC"
                    >
                      {
                        `Clients cumul\u00e9s : ${formatPercent(
                          hoveredPoint
                            .populationShare
                        )}`
                      }
                    </text>


                    <text
                      x="12"
                      y="40"
                      fontSize="11"
                      fontWeight="600"
                      fill="#E2E8F0"
                    >
                      {
                        `CA cumul\u00e9 : ${formatPercent(
                          hoveredPoint
                            .revenueShare
                        )}`
                      }
                    </text>


                    <text
                      x="12"
                      y="60"
                      fontSize="11"
                      fill="#94A3B8"
                    >
                      {
                        `\u00c9galit\u00e9 parfaite : ${formatPercent(
                          hoveredPoint
                            .equalityShare
                        )}`
                      }
                    </text>
                  </g>
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
            10
          }
          textAnchor="middle"
          className={
            styles.chartTick
          }
        >
          {
            "Part cumul\u00e9e des clients"
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
            "Part cumul\u00e9e du chiffre d\u2019affaires"
          }
        </text>


        <g
          transform={
            `translate(${padding.left + 10} 13)`
          }
          pointerEvents="none"
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
            Lorenz
          </text>

          <line
            x1="102"
            y1="0"
            x2="130"
            y2="0"
            stroke="currentColor"
            strokeWidth="2"
            strokeDasharray="7 6"
            opacity="0.42"
          />

          <text
            x="138"
            y="4"
            className={
              styles.chartTick
            }
          >
            {
              "\u00c9galit\u00e9 parfaite"
            }
          </text>
        </g>
      </svg>
    </div>
  );
}
