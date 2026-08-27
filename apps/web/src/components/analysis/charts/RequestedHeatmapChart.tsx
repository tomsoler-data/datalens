import type {
  ReportChartDatum,
} from "../../../app/types";

import styles from "../../../app/page.module.css";

import {
  datumLabel,
  datumNumber,
  formatChartNumber,
  formatNumber,
} from "../analysisPresentation";


export default function RequestedHeatmapChart({
  data,
  xLabel,
  yLabel,
}: {
  data:
    ReportChartDatum[];

  xLabel:
    string;

  yLabel:
    string;
}) {
  const cells =
    data
      .map(
        (
          datum
        ) => {
          const x =
            datumLabel(
              datum,
              "x"
            );

          const y =
            datumLabel(
              datum,
              "y"
            );

          const count =
            datumNumber(
              datum,
              "count"
            );


          if (
            x ===
              null ||
            y ===
              null ||
            count ===
              null
          ) {
            return null;
          }


          return {
            x,
            y,
            count,
          };
        }
      )
      .filter(
        (
          cell
        ): cell is {
          x:
            string;

          y:
            string;

          count:
            number;
        } =>
          cell !==
          null
      );


  if (
    cells.length ===
    0
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Aucune cellule de contingence
        exploitable.
      </div>
    );
  }


  const xValues =
    Array.from(
      new Set(
        cells.map(
          (
            cell
          ) =>
            cell.x
        )
      )
    );


  const yValues =
    Array.from(
      new Set(
        cells.map(
          (
            cell
          ) =>
            cell.y
        )
      )
    );


  const maxCount =
    Math.max(
      ...cells.map(
        (
          cell
        ) =>
          cell.count
      ),
      1
    );


  const width =
    860;


  /*
   * Une petite table de contingence ne doit pas être
   * comprimée visuellement.
   *
   * Le viewBox reste responsive, mais la matrice garde
   * suffisamment de hauteur pour rendre les écarts lisibles.
   */
  const height =
    Math.max(
      360,
      150 +
      yValues.length *
        88
    );


  const padding = {
    top:
      70,

    right:
      32,

    bottom:
      70,

    left:
      145,
  };


  const plotWidth =
    width -
    padding.left -
    padding.right;

  const plotHeight =
    height -
    padding.top -
    padding.bottom;


  const cellWidth =
    plotWidth /
    Math.max(
      xValues.length,
      1
    );

  const cellHeight =
    plotHeight /
    Math.max(
      yValues.length,
      1
    );


  const countMap =
    new Map<
      string,
      number
    >();


  for (
    const cell
    of cells
  ) {
    countMap.set(
      `${cell.x}|||${cell.y}`,
      cell.count
    );
  }


  const clamp01 = (
    value:
      number
  ) =>
    Math.max(
      0,
      Math.min(
        1,
        value
      )
    );


  const interpolateChannel = (
    from:
      number,

    to:
      number,

    progress:
      number
  ) =>
    Math.round(
      from +
      (
        to -
        from
      ) *
        progress
    );


  const interpolateRgb = (
    from:
      readonly [
        number,
        number,
        number,
      ],

    to:
      readonly [
        number,
        number,
        number,
      ],

    progress:
      number
  ) => {
    const t =
      clamp01(
        progress
      );


    return `rgb(${interpolateChannel(
      from[0],
      to[0],
      t
    )}, ${interpolateChannel(
      from[1],
      to[1],
      t
    )}, ${interpolateChannel(
      from[2],
      to[2],
      t
    )})`;
  };


  const heatmapCellStyle = (
    count:
      number,

    intensity:
      number
  ) => {
    if (
      count <=
      0
    ) {
      return {
        fill:
          "rgb(10, 18, 30)",

        stroke:
          "rgba(154, 197, 234, 0.22)",

        text:
          "rgba(190, 209, 229, 0.52)",
      };
    }


    const t =
      clamp01(
        intensity
      );


    const low = [
      28,
      76,
      116,
    ] as const;

    const middle = [
      88,
      154,
      210,
    ] as const;

    const high = [
      216,
      238,
      255,
    ] as const;


    const adjustedT =
      Math.pow(
        t,
        0.82
      );


    const fill =
      adjustedT <=
      0.5
        ? interpolateRgb(
            low,
            middle,
            adjustedT *
              2
          )
        : interpolateRgb(
            middle,
            high,
            (
              adjustedT -
              0.5
            ) *
              2
          );


    return {
      fill,

      stroke:
        adjustedT >=
        0.72
          ? "rgba(232, 244, 255, 0.82)"
          : "rgba(191, 222, 247, 0.56)",

      text:
        adjustedT >=
        0.74
          ? "#05101d"
          : "#f8fbff",
    };
  };


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
          `Table de contingence entre ${xLabel} et ${yLabel}`
        }
      >
        {
          xValues.map(
            (
              value,
              index
            ) => (
              <text
                key={
                  `x-${value}`
                }
                x={
                  padding.left +
                  (
                    index +
                    0.5
                  ) *
                    cellWidth
                }
                y={
                  padding.top -
                  22
                }
                textAnchor="middle"
                className={
                  styles.chartTick
                }
                style={{
                  fontSize:
                    14,

                  fontWeight:
                    650,

                  opacity:
                    0.92,
                }}
              >
                {
                  value
                }
              </text>
            )
          )
        }


        {
          yValues.map(
            (
              value,
              index
            ) => (
              <text
                key={
                  `y-${value}`
                }
                x={
                  padding.left -
                  18
                }
                y={
                  padding.top +
                  (
                    index +
                    0.5
                  ) *
                    cellHeight +
                  5
                }
                textAnchor="end"
                className={
                  styles.chartTick
                }
                style={{
                  fontSize:
                    14,

                  fontWeight:
                    650,

                  opacity:
                    0.92,
                }}
              >
                {
                  value
                }
              </text>
            )
          )
        }


        {
          yValues.flatMap(
            (
              yValue,
              yIndex
            ) =>
              xValues.map(
                (
                  xValue,
                  xIndex
                ) => {
                  const count =
                    countMap.get(
                      `${xValue}|||${yValue}`
                    ) ??
                    0;


                  const intensity =
                    count /
                    maxCount;


                  const cellStyle =
                    heatmapCellStyle(
                      count,
                      intensity
                    );


                  const observationLabel =
                    count ===
                    1
                      ?
                      "observation"
                      :
                      "observations";


                  return (
                    <g
                      key={
                        `${xValue}-${yValue}`
                      }
                    >
                      <rect
                        x={
                          padding.left +
                          xIndex *
                            cellWidth +
                          3
                        }
                        y={
                          padding.top +
                          yIndex *
                            cellHeight +
                          3
                        }
                        width={
                          Math.max(
                            0,
                            cellWidth -
                            6
                          )
                        }
                        height={
                          Math.max(
                            0,
                            cellHeight -
                            6
                          )
                        }
                        rx="12"
                        fill={
                          cellStyle.fill
                        }
                        stroke={
                          cellStyle.stroke
                        }
                        strokeWidth="1.5"
                      >
                        <title>
                          {
                            `${xLabel}: ${xValue} \u00b7 ${yLabel}: ${yValue} \u00b7 ${formatNumber(
                              count
                            )} ${observationLabel}`
                          }
                        </title>
                      </rect>


                      <text
                        x={
                          padding.left +
                          (
                            xIndex +
                            0.5
                          ) *
                            cellWidth
                        }
                        y={
                          padding.top +
                          (
                            yIndex +
                            0.5
                          ) *
                            cellHeight +
                          6
                        }
                        textAnchor="middle"
                        className={
                          styles.chartTick
                        }
                        style={{
                          pointerEvents:
                            "none",

                          fill:
                            cellStyle.text,

                          fontSize:
                            17,

                          fontWeight:
                            800,

                          textShadow:
                            intensity >=
                            0.78
                              ? "none"
                              : "0 1px 2px rgba(2, 8, 18, 0.45)",
                        }}
                      >
                        {
                          formatChartNumber(
                            count
                          )
                        }
                      </text>
                    </g>
                  );
                }
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
            14
          }
          textAnchor="middle"
          className={
            styles.chartTick
          }
          style={{
            fontSize:
              13,

            fontWeight:
              700,

            opacity:
              0.86,
          }}
        >
          {
            xLabel
          }
        </text>


        <text
          x="22"
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
            `rotate(-90 22 ${
              padding.top +
              plotHeight /
                2
            })`
          }
          style={{
            fontSize:
              13,

            fontWeight:
              700,

            opacity:
              0.86,
          }}
        >
          {
            yLabel
          }
        </text>
      </svg>
    </div>
  );
}
