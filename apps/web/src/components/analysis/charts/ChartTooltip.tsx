export function clampChartTooltipPosition(
  anchorX: number,
  anchorY: number,
  tooltipWidth: number,
  tooltipHeight: number,
  svgWidth: number,
  svgHeight: number
): {
  x: number;
  y: number;
} {
  const margin =
    12;

  const preferredX =
    anchorX +
    14;

  const fallbackX =
    anchorX -
    tooltipWidth -
    14;

  const x =
    preferredX +
      tooltipWidth <=
    svgWidth -
      margin
      ? preferredX
      : Math.max(
          margin,
          fallbackX
        );


  const y =
    Math.max(
      margin,
      Math.min(
        svgHeight -
          tooltipHeight -
          margin,
        anchorY -
          tooltipHeight /
            2
      )
    );


  return {
    x,
    y,
  };
}


export function SvgChartTooltip({
  x,
  y,
  lines,
  width =
    220,
}: {
  x:
    number;

  y:
    number;

  lines:
    string[];

  width?:
    number;
}) {
  const padding =
    12;

  const lineHeight =
    18;

  const height =
    padding *
      2 +
    lines.length *
      lineHeight;


  return (
    <g
      transform={
        `translate(${x} ${y})`
      }
      pointerEvents="none"
      aria-hidden="true"
    >
      <rect
        width={
          width
        }
        height={
          height
        }
        rx="10"
        fill="#091321"
        stroke="rgba(164, 199, 255, 0.28)"
        strokeWidth="1"
      />

      {
        lines.map(
          (
            line,
            index
          ) => (
            <text
              key={
                `${index}-${line}`
              }
              x={
                padding
              }
              y={
                padding +
                13 +
                index *
                  lineHeight
              }
              fill="currentColor"
              style={{
                fontSize:
                  index ===
                    0
                    ? "12px"
                    : "11px",

                fontWeight:
                  index ===
                    0
                    ? 700
                    : 500,

                opacity:
                  index ===
                    0
                    ? 0.96
                    : 0.78,
              }}
            >
              {
                line
              }
            </text>
          )
        )
      }
    </g>
  );
}
