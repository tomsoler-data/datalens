import type {
  ReportChartDatum,
} from "../../app/types";


const FRIENDLY_VARIABLE_LABELS:
  Record<string, string> = {
    "Population using at least basic drinking-water services (%)":
      "Accès basique à l’eau potable",

    "Population using safely managed drinking-water services (%)":
      "Accès sécurisé à l’eau potable",

    "Mortality rate attributed to exposure to unsafe WASH services":
      "Mortalité liée à l’eau et aux services WASH",

    "Political Stability":
      "Stabilité politique",

    "Political_Stability":
      "Stabilité politique",

    "WASH deaths":
      "Décès liés aux services WASH",

    Population:
      "Population",

    Year:
      "Année",

    Country:
      "Pays",

    Granularity:
      "Zone",

    "REGION (DISPLAY)":
      "Région",

    "COUNTRY (DISPLAY)":
      "Pays",

    Age:
      "Âge",

    Basket:
      "Panier",

    age_at_first_purchase:
      "Âge au premier achat",

    total_spend:
      "Montant total des achats",

    purchase_sessions:
      "Fréquence d’achat",

    average_basket:
      "Panier moyen",

    median_basket:
      "Panier médian",

    sum_price:
      "Montant agrégé",

    price:
      "Prix",

    event_count:
      "Nombre d’événements",

    categ:
      "Catégorie",

    category:
      "Catégorie",

    gender:
      "Genre",

    customer_id:
      "Client",

    event_time:
      "Date d’achat",

    month:
      "Mois",

    gross_amount:
      "Chiffre d’affaires",

    sum_gross_amount:
      "Chiffre d’affaires",
  };

export function formatNumber(
  value: number
): string {
  return new Intl
    .NumberFormat(
      "fr-FR"
    )
    .format(
      value
    );
}

export function formatDecimal(
  value: number
): string {
  if (
    Math.abs(value) <
      0.001 &&
    value !==
      0
  ) {
    return value
      .toExponential(
        2
      );
  }


  return new Intl
    .NumberFormat(
      "fr-FR",
      {
        maximumFractionDigits:
          3,
      }
    )
    .format(
      value
    );
}

export function formatChartNumber(
  value: number
): string {
  const absolute =
    Math.abs(
      value
    );


  if (
    absolute >=
      1000
  ) {
    return new Intl
      .NumberFormat(
        "fr-FR",
        {
          notation:
            "compact",

          maximumFractionDigits:
            1,
        }
      )
      .format(
        value
      );
  }


  return new Intl
    .NumberFormat(
      "fr-FR",
      {
        maximumFractionDigits:
          2,
      }
    )
    .format(
      value
    );
}

export function formatAxisNumber(
  value: number
): string {
  const absolute =
    Math.abs(
      value
    );


  if (
    absolute >=
    1_000_000
  ) {
    return new Intl
      .NumberFormat(
        "fr-FR",
        {
          notation:
            "compact",

          maximumFractionDigits:
            1,
        }
      )
      .format(
        value
      );
  }


  return new Intl
    .NumberFormat(
      "fr-FR",
      {
        maximumFractionDigits:
          absolute <
            10
            ? 2
            : 1,
      }
    )
    .format(
      value
    );
}

export function formatPercent(
  value: number
): string {
  return new Intl
    .NumberFormat(
      "fr-FR",
      {
        style:
          "percent",

        maximumFractionDigits:
          1,
      }
    )
    .format(
      value
    );
}

export function formatTemporalDisplayValue(
  value:
    unknown
): string {
  if (
    value ===
      null ||
    value ===
      undefined
  ) {
    return "—";
  }


  if (
    typeof value ===
      "number"
  ) {
    if (
      Number.isInteger(
        value
      ) &&
      value >=
        1000 &&
      value <=
        9999
    ) {
      return String(
        value
      );
    }


    return formatChartNumber(
      value
    );
  }


  const raw =
    String(
      value
    ).trim();


  if (
    !raw
  ) {
    return "—";
  }


  if (
    /^\d{4}$/.test(
      raw
    )
  ) {
    return raw;
  }


  const isoDateMatch =
    raw.match(
      /^(\d{4})-(\d{2})-(\d{2})(?:[T\s].*)?$/
    );


  if (
    isoDateMatch
  ) {
    const year =
      Number(
        isoDateMatch[
          1
        ]
      );

    const month =
      Number(
        isoDateMatch[
          2
        ]
      );

    const day =
      Number(
        isoDateMatch[
          3
        ]
      );


    const calendarDate =
      new Date(
        Date.UTC(
          year,
          month -
            1,
          day
        )
      );


    return new Intl.DateTimeFormat(
      "fr-FR",
      {
        day:
          "2-digit",

        month:
          "short",

        year:
          "numeric",

        timeZone:
          "UTC",
      }
    ).format(
      calendarDate
    );
  }


  const parsed =
    new Date(
      raw
    );


  if (
    Number.isNaN(
      parsed.getTime()
    )
  ) {
    return raw;
  }


  return new Intl.DateTimeFormat(
    "fr-FR",
    {
      day:
        "2-digit",

      month:
        "short",

      year:
        "numeric",

      timeZone:
        "UTC",
    }
  ).format(
    parsed
  );
}

export function friendlyVariableLabel(
  value: string
): string {
  const direct =
    FRIENDLY_VARIABLE_LABELS[
      value
    ];


  if (
    direct
  ) {
    return direct;
  }


  return value
    .replace(
      /\s*\(%\)\s*$/i,
      ""
    )
    .replace(
      /_/g,
      " "
    )
    .trim();
}

export function metricNumber(
  metrics:
    Record<
      string,
      unknown
    >,

  key: string
): number | null {
  const value =
    metrics[
      key
    ];


  return (
    typeof value ===
      "number" &&
    Number.isFinite(
      value
    )
  )
    ? value
    : null;
}

export function metricString(
  metrics:
    Record<
      string,
      unknown
    >,

  key: string
): string | null {
  const value =
    metrics[
      key
    ];


  return typeof value ===
    "string"
      ? value
      : null;
}

export function datumNumber(
  datum:
    ReportChartDatum,

  key: string
): number | null {
  const value =
    datum[
      key
    ];


  return (
    typeof value ===
      "number" &&
    Number.isFinite(
      value
    )
  )
    ? value
    : null;
}

export function datumLabel(
  datum:
    ReportChartDatum,

  key: string
): string | null {
  const value =
    datum[
      key
    ];


  if (
    typeof value ===
    "string"
  ) {
    const trimmed =
      value.trim();


    return trimmed
      ? trimmed
      : null;
  }


  if (
    typeof value ===
      "number" &&
    Number.isFinite(
      value
    )
  ) {
    return formatDecimal(
      value
    );
  }


  return null;
}


export function analysisKindLabel(
  value: string
): string {
  switch (
    value
  ) {
    case "quantitative":
      return "Quantitative";

    case "temporal":
      return "Temporelle";

    case "categorical":
      return "Catégorielle";

    case "boolean":
      return "Booléenne";

    default:
      return "À déterminer";
  }
}
