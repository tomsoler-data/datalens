from __future__ import annotations


import re

from datetime import (
    datetime,
)

from io import (
    BytesIO,
)

from typing import (
    Any,
)

from xml.sax.saxutils import (
    escape,
)


from reportlab.graphics.shapes import (
    Drawing,
    Line,
    PolyLine,
    Rect,
    String,
)

from reportlab.lib import (
    colors,
)

from reportlab.lib.enums import (
    TA_LEFT,
)

from reportlab.lib.pagesizes import (
    A4,
)

from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)

from reportlab.lib.units import (
    mm,
)

from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


from app.reporting.analysis_artifact_store import (
    list_analysis_artifacts,
)


from app.reporting.report_selection_store import (
    ReportSelectionDetailResponse,
)


# ============================================================
# UNRESOLVED DOCUMENT REQUESTS
# ============================================================

def unresolved_requested_analyses_for_workflow(
    workflow_id: str,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """
    Read persisted non-executed documentary requests.

    These objects are lifecycle records only. They are not
    analytical findings and are never read from report
    selection.
    """

    normalized_workflow_id = str(
        workflow_id
        or
        ""
    ).strip()


    if not (
        normalized_workflow_id
    ):
        return []


    records = (
        list_analysis_artifacts(
            workflow_id=
                normalized_workflow_id
        )
    )


    unresolved: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for record in records:
        if (
            record.source_type
            !=
            "document_request"
        ):
            continue


        if (
            record.executed
        ):
            continue


        payload = (
            record.pipeline_payload
            if isinstance(
                record.pipeline_payload,
                dict,
            )
            else
            {}
        )


        if (
            str(
                payload.get(
                    "artifact_kind",
                    "",
                )
            )
            .strip()
            !=
            "requested_analysis_lifecycle"
        ):
            continue


        lifecycle = (
            payload.get(
                "request_lifecycle"
            )
        )


        if not isinstance(
            lifecycle,
            dict,
        ):
            continue


        execution_status = str(
            lifecycle.get(
                "execution_status",
                "",
            )
            or
            ""
        ).strip().lower()


        if (
            execution_status
            !=
            "not_executed"
        ):
            continue


        request_order_raw = (
            lifecycle.get(
                "request_order"
            )
        )


        try:
            request_order = int(
                request_order_raw
            )

        except (
            TypeError,
            ValueError,
        ):
            request_order = None


        unresolved.append(
            {
                "analysis_id":
                    record.analysis_id,

                "request_id":
                    lifecycle.get(
                        "request_id"
                    ),

                "request_text":
                    (
                        lifecycle.get(
                            "request_text"
                        )
                        or
                        record.objective
                    ),

                "request_order":
                    request_order,

                "plan_status":
                    lifecycle.get(
                        "plan_status"
                    ),

                "execution_status":
                    execution_status,

                "warnings":
                    list(
                        lifecycle.get(
                            "warnings"
                        )
                        or
                        []
                    ),

                "limitations":
                    list(
                        lifecycle.get(
                            "limitations"
                        )
                        or
                        []
                    ),

                "source_filename":
                    lifecycle.get(
                        "source_filename"
                    ),

                "source_locator":
                    lifecycle.get(
                        "source_locator"
                    ),

                "page_number":
                    lifecycle.get(
                        "page_number"
                    ),

                "evidence_quote":
                    lifecycle.get(
                        "evidence_quote"
                    ),

                "created_at_utc":
                    record.created_at_utc,
            }
        )


    unresolved.sort(
        key=lambda item:
            (
                (
                    item[
                        "request_order"
                    ]
                    if isinstance(
                        item.get(
                            "request_order"
                        ),
                        int,
                    )
                    else
                    10 ** 9
                ),
                str(
                    item.get(
                        "created_at_utc",
                        "",
                    )
                    or
                    ""
                ),
                str(
                    item.get(
                        "request_id",
                        "",
                    )
                    or
                    ""
                ),
                str(
                    item.get(
                        "analysis_id",
                        "",
                    )
                    or
                    ""
                ),
            )
    )


    return unresolved


def unresolved_request_status_label(
    value: object,
) -> str:
    normalized = str(
        value
        or
        ""
    ).strip().lower()


    labels = {
        "ambiguous":
            "Ambigu\u00eb",

        "blocked":
            "Bloqu\u00e9e",

        "not_executed":
            "Non ex\u00e9cut\u00e9e",
    }


    return (
        labels.get(
            normalized
        )
        or
        normalized.replace(
            "_",
            " ",
        ).capitalize()
        or
        "Non ex\u00e9cut\u00e9e"
    )


def unresolved_request_reason_lines(
    request: dict[
        str,
        Any,
    ],
) -> list[
    str
]:
    warnings = (
        request.get(
            "warnings"
        )
        or
        []
    )


    limitations = (
        request.get(
            "limitations"
        )
        or
        []
    )


    source_values = (
        warnings
        if warnings
        else
        limitations
    )


    rendered: list[
        str
    ] = []


    seen: set[
        str
    ] = set()


    for value in source_values:
        text = clean_text(
            value
        )


        if not (
            text
        ):
            continue


        text = clean_text(
            translate_executor_text(
                text
            )
        )


        if (
            not text
            or
            text in seen
        ):
            continue


        seen.add(
            text
        )


        rendered.append(
            text
        )


        if (
            len(
                rendered
            )
            >=
            2
        ):
            break


    if not (
        rendered
    ):
        rendered.append(
            (
                "Aucun motif d\u00e9taill\u00e9 "
                "n'a \u00e9t\u00e9 persist\u00e9 "
                "pour cette demande."
            )
        )


    return rendered


def unresolved_request_source_text(
    request: dict[
        str,
        Any,
    ],
) -> str:
    filename = clean_text(
        request.get(
            "source_filename"
        )
    )


    locator = clean_text(
        request.get(
            "source_locator"
        )
    )


    page_number = (
        request.get(
            "page_number"
        )
    )


    if (
        filename
        and
        page_number
        is not None
    ):
        return (
            f"{filename} - page {page_number}"
        )


    if (
        filename
        and
        locator
    ):
        return (
            f"{filename} - {locator}"
        )


    return (
        filename
        or
        locator
    )


def build_unresolved_requested_section(
    *,
    workflow_id: str,
    styles: dict[
        str,
        Any,
    ],
) -> list[
    Any
]:
    requests = (
        unresolved_requested_analyses_for_workflow(
            workflow_id
        )
    )


    if not (
        requests
    ):
        return []


    flowables: list[
        Any
    ] = []


    ambiguous_count = sum(
        1

        for request in requests

        if (
            clean_text(
                request.get(
                    "plan_status"
                )
            )
            .lower()
            ==
            "ambiguous"
        )
    )


    blocked_count = sum(
        1

        for request in requests

        if (
            clean_text(
                request.get(
                    "plan_status"
                )
            )
            .lower()
            ==
            "blocked"
        )
    )


    status_style = ParagraphStyle(
        "SelectedReportUnresolvedStatus",
        parent=
            styles[
                "small"
            ],
        fontName=
            "Helvetica-Bold",
        fontSize=
            7.5,
        leading=
            9,
        textColor=
            WARNING,
    )


    flowables.append(
        Paragraph(
            "Demandes non ex\u00e9cut\u00e9es",
            styles[
                "h1"
            ],
        )
    )


    flowables.append(
        Paragraph(
            (
                f"{len(requests)} demande(s) issue(s) des "
                "documents n'ont pas produit de r\u00e9sultat "
                "analytique. Elles restent enregistr\u00e9es "
                "c\u00f4t\u00e9 serveur afin de conserver la "
                "tra\u00e7abilit\u00e9 de la demande sans les "
                "pr\u00e9senter comme des analyses ex\u00e9cut\u00e9es."
            ),
            styles[
                "body"
            ],
        )
    )


    flowables.append(
        Spacer(
            1,
            5,
        )
    )


    if (
        ambiguous_count
        or
        blocked_count
    ):
        summary_parts: list[
            str
        ] = []


        if (
            ambiguous_count
        ):
            summary_parts.append(
                (
                    f"{ambiguous_count} "
                    "ambigu\u00eb(s)"
                )
            )


        if (
            blocked_count
        ):
            summary_parts.append(
                (
                    f"{blocked_count} "
                    "bloqu\u00e9e(s)"
                )
            )


        flowables.append(
            Paragraph(
                (
                    "<b>Statuts :</b> "
                    +
                    escape(
                        " \u00b7 ".join(
                            summary_parts
                        )
                    )
                ),
                styles[
                    "body"
                ],
            )
        )


        flowables.append(
            Spacer(
                1,
                8,
            )
        )


    for (
        index,
        request,
    ) in enumerate(
        requests,
        start=1,
    ):
        title = clean_text(
            request.get(
                "request_text"
            )
        )


        status = (
            unresolved_request_status_label(
                request.get(
                    "plan_status"
                )
            )
        )


        reason_lines = (
            unresolved_request_reason_lines(
                request
            )
        )


        source_text = (
            unresolved_request_source_text(
                request
            )
        )


        request_order = (
            request.get(
                "request_order"
            )
        )


        eyebrow = (
            "DEMANDE DOCUMENTAIRE"
        )


        if isinstance(
            request_order,
            int,
        ):
            eyebrow += (
                f" \u00b7 DEMANDE {request_order}"
            )


        eyebrow += (
            f" \u00b7 {status.upper()}"
        )


        block: list[
            Any
        ] = [
            Paragraph(
                escape(
                    eyebrow
                ),
                status_style,
            ),

            Spacer(
                1,
                3,
            ),

            Paragraph(
                escape(
                    title
                    or
                    "Demande sans intitul\u00e9"
                ),
                styles[
                    "h2"
                ],
            ),

            Spacer(
                1,
                3,
            ),

            Paragraph(
                (
                    "<b>Statut :</b> "
                    +
                    escape(
                        status
                    )
                ),
                styles[
                    "body"
                ],
            ),
        ]


        for (
            reason_index,
            reason,
        ) in enumerate(
            reason_lines,
            start=1,
        ):
            label = (
                "Motif"
                if reason_index
                ==
                1
                else
                "Pr\u00e9cision"
            )


            block.append(
                Paragraph(
                    (
                        f"<b>{label} :</b> "
                        +
                        escape(
                            reason
                        )
                    ),
                    styles[
                        "body"
                    ],
                )
            )


        if (
            source_text
        ):
            block.extend(
                [
                    Spacer(
                        1,
                        3,
                    ),

                    Paragraph(
                        (
                            "Demande issue du document : "
                            +
                            escape(
                                source_text
                            )
                        ),
                        styles[
                            "small"
                        ],
                    ),
                ]
            )


        flowables.append(
            KeepTogether(
                block
            )
        )


        if (
            index
            <
            len(
                requests
            )
        ):
            flowables.append(
                Spacer(
                    1,
                    14,
                )
            )


    return flowables


# ============================================================
# VERSION
# ============================================================

SELECTED_REPORT_PDF_RULE_VERSION = (
    "selected_report_pdf_v0.5"
)


# ============================================================
# COLORS
# ============================================================

INK = colors.HexColor(
    "#162033"
)

MUTED = colors.HexColor(
    "#5F6B7A"
)

ACCENT = colors.HexColor(
    "#356FD6"
)

ACCENT_DARK = colors.HexColor(
    "#204A91"
)

ACCENT_LIGHT = colors.HexColor(
    "#EAF1FF"
)

GRID = colors.HexColor(
    "#D9E1EC"
)

PANEL = colors.HexColor(
    "#F6F8FB"
)

PANEL_BLUE = colors.HexColor(
    "#F1F6FF"
)

SUCCESS = colors.HexColor(
    "#147D64"
)

SUCCESS_LIGHT = colors.HexColor(
    "#E8F5F1"
)

WARNING = colors.HexColor(
    "#9A6412"
)

WHITE = colors.white


# ============================================================
# TEXT / VALUE HELPERS
# ============================================================

def normalize_pdf_ligatures(
    value: str,
) -> str:
    """
    Normalize typographic ligatures that are not supported by
    the built-in ReportLab Helvetica fonts.

    This is presentation-only normalization. Server-owned
    source text and analytical artifacts remain unchanged.
    """

    replacements = {
        "\ufb00":
            "ff",

        "\ufb01":
            "fi",

        "\ufb02":
            "fl",

        "\ufb03":
            "ffi",

        "\ufb04":
            "ffl",

        "\ufb05":
            "st",

        "\ufb06":
            "st",
    }


    normalized = str(
        value
        or
        ""
    )


    for (
        ligature,
        replacement,
    ) in replacements.items():
        normalized = normalized.replace(
            ligature,
            replacement,
        )


    return normalized


def clean_text(
    value: object,
) -> str:
    text = normalize_pdf_ligatures(
        str(
            value
            or
            ""
        )
    )


    replacements = {
        "\u2018":
            "'",

        "\u2019":
            "'",

        "\u201c":
            '"',

        "\u201d":
            '"',

        "\u2013":
            "-",

        "\u2014":
            "-",

        "\u202f":
            " ",

        "\u00a0":
            " ",

        "\u2192":
            "->",

        "\u2197":
            "->",

        "\u0153":
            "oe",

        "\u0152":
            "OE",
    }


    for (
        source,
        target,
    ) in replacements.items():
        text = text.replace(
            source,
            target,
        )


    return " ".join(
        text.split()
    )


def safe_number(
    value: object,
) -> (
    float
    | None
):
    if isinstance(
        value,
        bool,
    ):
        return None


    if isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        number = float(
            value
        )


        if (
            number ==
            number

            and

            number
            not in {
                float(
                    "inf"
                ),
                float(
                    "-inf"
                ),
            }
        ):
            return number


    return None


KNOWN_TRANSLATIONS = {
    (
        "DataLens calculated the sample size, mean, median, "
        "dispersion and quartiles for each group."
    ):
        (
            "DataLens a calculé l'effectif, la moyenne, la médiane, "
            "la dispersion et les quartiles pour chaque groupe."
        ),

    (
        "This execution compares group distributions descriptively. "
        "No inferential group-comparison test has been applied yet."
    ):
        (
            "Cette analyse compare les distributions des groupes "
            "de manière descriptive. Aucun test inférentiel de "
            "comparaison de groupes n'a été appliqué."
        ),

    (
        "This AI-native time-series execution is descriptive. "
        "It does not fit a forecasting, causal or inferential "
        "time-series model."
    ):
        (
            "Cette analyse temporelle est descriptive. Aucun modèle "
            "de prévision, causal ou inférentiel n'a été ajusté."
        ),

    "No inferential temporal model was executed.":
        "Aucun modèle temporel inférentiel n'a été exécuté.",
}


def paragraph_text(
    value: object,
) -> str:
    return escape(
        clean_text(
            value
        )
    )


def format_number(
    value: object,
    *,
    decimals: int = 2,
) -> str:
    """French display format used in the report body."""
    number = (
        safe_number(
            value
        )
    )


    if (
        number is None
    ):
        normalized = clean_text(
            value
        )


        labels = {
            "mean":
                "Moyenne",

            "median":
                "Médiane",

            "sum":
                "Somme",

            "min":
                "Minimum",

            "max":
                "Maximum",

            "count":
                "Effectif",

            "distinct_count":
                "Valeurs distinctes",
        }


        return (
            labels.get(
                normalized,
                normalized,
            )
        )


    if float(
        number
    ).is_integer():
        return (
            f"{int(number):,}"
            .replace(
                ",",
                " ",
            )
        )


    return (
        f"{number:,.{decimals}f}"
        .replace(
            ",",
            "\u00a0",
        )
        .replace(
            ".",
            ",",
        )
        .replace(
            "\u00a0",
            " ",
        )
    )


def translate_executor_text(
    value: object,
) -> str:
    text = clean_text(
        value
    )


    known = (
        KNOWN_TRANSLATIONS.get(
            text
        )
    )


    if (
        known is not None
    ):
        return known


    group_summary_match = re.fullmatch(
        (
            r"(\d+)\s+groupe\(s\)\s+ont été comparés pour\s+(.+?)\."
        ),
        text,
        flags=re.IGNORECASE,
    )


    if (
        group_summary_match
    ):
        count = group_summary_match.group(
            1
        )

        metric = clean_text(
            group_summary_match.group(
                2
            )
        )


        return (
            f"{count} groupes ont été comparés pour {metric}."
        )


    ranking_summary_match = re.fullmatch(
        (
            r"(\d+)\s+résultat\(s\)\s+ont été conservés "
            r"après classement décroissant\."
        ),
        text,
        flags=re.IGNORECASE,
    )


    if (
        ranking_summary_match
    ):
        count = ranking_summary_match.group(
            1
        )


        return (
            f"{count} résultats ont été retenus après classement décroissant."
        )


    ranking_match = re.fullmatch(
        (
            r"(?:First result|Premier résultat)\s*:\s*"
            r"(.+?)\s*=\s*"
            r"(-?\d+(?:\.\d+)?)\."
        ),
        text,
        flags=re.IGNORECASE,
    )


    if (
        ranking_match
    ):
        label = clean_text(
            ranking_match.group(
                1
            )
        )

        value_number = float(
            ranking_match.group(
                2
            )
        )


        return (
            "Premier résultat : "
            f"{label} - "
            f"{format_number(value_number)}."
        )


    return text


def pdf_warning_messages(
    *,
    family: str,
    warnings: object,
) -> list[
    str
]:
    """
    Convert deterministic executor warnings into a concise
    business-readable PDF presentation.

    Original warnings remain untouched in server-owned
    analytical artifacts.
    """

    if not isinstance(
        warnings,
        list,
    ):
        return []


    raw_warnings = [
        clean_text(
            warning
        )

        for warning in warnings

        if clean_text(
            warning
        )
    ]


    if not raw_warnings:
        return []


    lowered = [
        warning.casefold()

        for warning in raw_warnings
    ]


    output: list[
        str
    ] = []


    # ========================================================
    # CATEGORICAL ASSOCIATION / INDEPENDENCE
    # ========================================================

    independence_detected = any(
        (
            "independence assumption"
            in warning
        )
        or
        (
            "ind\u00e9pendance"
            in warning
        )
        for warning in lowered
    )


    repeated_client_detected = any(
        (
            "achats r\u00e9p\u00e9t\u00e9s"
            in warning
        )
        or
        (
            "repeated observations"
            in warning
        )
        for warning in lowered
    )


    if (
        family
        ==
        "categorical_association"
        and
        (
            independence_detected
            or
            repeated_client_detected
        )
    ):
        output.append(
            (
                "Plusieurs achats peuvent appartenir au m\u00eame "
                "client. Les observations ne sont donc pas "
                "consid\u00e9r\u00e9es comme ind\u00e9pendantes et "
                "aucun test du Khi\u00b2 n'est interpr\u00e9t\u00e9 "
                "automatiquement."
            )
        )


    # ========================================================
    # QUANTITATIVE ASSOCIATION / OUTLIERS
    # ========================================================

    outliers_detected = any(
        (
            "potential outliers"
            in warning
        )
        or
        (
            "1.5 \u00d7 iqr"
            in warning
        )
        or
        (
            "1,5 \u00d7 iqr"
            in warning
        )
        for warning in lowered
    )


    if (
        family
        ==
        "quantitative_association"
        and
        outliers_detected
    ):
        output.append(
            (
                "Des valeurs potentiellement atypiques ont "
                "\u00e9t\u00e9 d\u00e9tect\u00e9es par le diagnostic "
                "1,5 \u00d7 IQR. Elles n'ont pas \u00e9t\u00e9 "
                "supprim\u00e9es."
            )
        )


    # ========================================================
    # QUANTITATIVE ASSOCIATION / RELATIONSHIP SHAPE
    # ========================================================

    unclear_shape_detected = any(
        (
            "do not identify a sufficiently clear linear"
            in warning
        )
        or
        (
            "sufficiently clear linear or monotonic"
            in warning
        )
        for warning in lowered
    )


    if (
        family
        ==
        "quantitative_association"
        and
        unclear_shape_detected
    ):
        output.append(
            (
                "Les diagnostics exploratoires ne montrent pas "
                "une forme de relation suffisamment claire pour "
                "privil\u00e9gier automatiquement une relation "
                "lin\u00e9aire ou monotone."
            )
        )


    # ========================================================
    # QUANTITATIVE ASSOCIATION / DESCRIPTIVE FALLBACK
    # ========================================================

    descriptive_correlation_detected = any(
        (
            "did not select a sufficiently defensible correlation test"
            in warning
        )
        or
        (
            "coefficients de pearson et de spearman"
            in warning
        )
        or
        (
            "aucune p-value"
            in warning
        )
        or
        (
            "data-driven selection"
            in warning
        )
        for warning in lowered
    )


    if (
        family
        ==
        "quantitative_association"
        and
        descriptive_correlation_detected
    ):
        output.append(
            (
                "L'analyse reste exploratoire : aucun test de "
                "corr\u00e9lation suffisamment d\u00e9fendable "
                "n'a \u00e9t\u00e9 s\u00e9lectionn\u00e9 "
                "automatiquement. Les coefficients de Pearson "
                "et de Spearman sont descriptifs et aucune "
                "p-value n'est interpr\u00e9t\u00e9e."
            )
        )


    # ========================================================
    # GROUP COMPARISON
    # ========================================================

    descriptive_group_detected = any(
        (
            "compares group distributions descriptively"
            in warning
        )
        or
        (
            "comparaisons sont donc descriptives"
            in warning
        )
        for warning in lowered
    )


    if (
        family
        ==
        "group_comparison"
        and
        descriptive_group_detected
    ):
        output.append(
            (
                "Les distributions des groupes sont compar\u00e9es "
                "de mani\u00e8re descriptive. Aucun test "
                "inf\u00e9rentiel de comparaison de groupes "
                "n'a \u00e9t\u00e9 appliqu\u00e9."
            )
        )


    repeated_group_detected = any(
        (
            "repeated observations over time"
            in warning
        )
        or
        (
            "achats r\u00e9p\u00e9t\u00e9s"
            in warning
        )
        for warning in lowered
    )


    if (
        family
        ==
        "group_comparison"
        and
        repeated_group_detected
    ):
        output.append(
            (
                "Des observations r\u00e9p\u00e9t\u00e9es ont "
                "\u00e9t\u00e9 d\u00e9tect\u00e9es pour certains "
                "clients. Un test supposant des groupes "
                "ind\u00e9pendants n\u00e9cessiterait donc des "
                "v\u00e9rifications suppl\u00e9mentaires sur le "
                "plan de d\u00e9pendance."
            )
        )


    # ========================================================
    # GENERIC FALLBACK
    #
    # Never silently discard an unknown executor warning.
    # ========================================================

    if not output:
        for warning in raw_warnings:
            rendered = translate_executor_text(
                warning
            )


            if (
                rendered
                and
                rendered
                not in output
            ):
                output.append(
                    rendered
                )


    # Static reports should remain readable.
    # Original warnings remain available in technical artifacts.
    return output[
        :
        3
    ]


def truncate(
    value: object,
    max_length: int = 42,
) -> str:
    text = clean_text(
        value
    )


    if (
        len(
            text
        )
        <=
        max_length
    ):
        return text


    return (
        text[
            :
            max_length
            -
            1
        ]
        .rstrip()
        +
        "..."
    )


def source_type_label(
    source_type: str,
) -> str:
    if (
        source_type ==
        "initial_request"
    ):
        return (
            "Demande initiale"
        )


    if (
        source_type ==
        "follow_up_prompt"
    ):
        return (
            "Question de suivi"
        )


    if (
        source_type ==
        "document_request"
    ):
        return (
            "Demande du document"
        )


    return (
        "Analyse automatique"
    )


def display_source_filename(
    value: object,
) -> str:
    from urllib.parse import unquote_plus

    raw = clean_text(value)
    if not raw:
        return ""

    try:
        return clean_text(
            unquote_plus(raw)
        )
    except Exception:
        return raw


def normalize_business_summary(
    value: object,
) -> str:
    rendered = translate_executor_text(
        value
    )

    transaction_match = re.fullmatch(
        (
            r"(\d+)\s+événement\(s\)\s+"
            r"transactionnel\(s\)\s+sont présents "
            r"dans le dataset préparé\."
        ),
        rendered,
        flags=re.IGNORECASE,
    )

    if transaction_match:
        count = int(
            transaction_match.group(1)
        )
        return (
            f"{format_number(count)} transactions "
            "sont présentes dans le jeu de données préparé."
        )

    product_match = re.fullmatch(
        (
            r"(\d+)\s+occurrence\(s\)\s+produit "
            r"sont observées dans les événements "
            r"transactionnels\."
        ),
        rendered,
        flags=re.IGNORECASE,
    )

    if product_match:
        count = int(
            product_match.group(1)
        )
        return (
            f"{format_number(count)} occurrences de produits "
            "sont observées dans les transactions."
        )

    distinct_product_match = re.fullmatch(
        (
            r"Elles concernent\s+(\d+)\s+référence\(s\)\s+"
            r"produit distincte\(s\)\."
        ),
        rendered,
        flags=re.IGNORECASE,
    )

    if distinct_product_match:
        count = int(
            distinct_product_match.group(1)
        )
        return (
            f"Elles concernent {format_number(count)} "
            "références produit distinctes."
        )

    return rendered


def family_label(
    family: str,
) -> str:
    labels = {
        "group_comparison":
            "Comparaison de groupes",

        "ranking":
            "Classement",

        "aggregation":
            "Agrégation",

        "aggregate_breakdown":
            "R\u00e9partition agr\u00e9g\u00e9e",

        "categorical_breakdown":
            "R\u00e9partition par cat\u00e9gorie",

        "inequality":
            "Analyse de concentration",

        "quantitative_association":
            "Association quantitative",

        "categorical_association":
            "Association catégorielle",

        "distribution":
            "Distribution",

        "time_series":
            "Série temporelle",

        "descriptive_metric":
            "Indicateur descriptif",
    }


    return (
        labels.get(
            family,
            clean_text(
                family
            )
            or
            "Analyse",
        )
    )


def stage_label(
    stage: str,
) -> str:
    labels = {
        "import":
            "Import",

        "understand":
            "Comprendre",

        "quality":
            "Qualité",

        "clean":
            "Nettoyer",

        "transform":
            "Transformer",

        "combine":
            "Combiner",

        "validate":
            "Finaliser",
    }


    return (
        labels.get(
            stage,
            clean_text(
                stage
            ).capitalize(),
        )
    )


def status_label(
    status: str,
    *,
    stage: str = "",
    required: bool | None = None,
    materialized: bool | None = None,
) -> str:
    normalized_status = clean_text(status).lower()
    normalized_stage = clean_text(stage).lower()

    if normalized_status == "skipped":
        return "Non requis"

    if (
        normalized_status == "passed"
        and normalized_stage in {"clean", "transform", "combine"}
    ):
        if materialized is True:
            return "Appliqué"
        if required is False:
            return "Non requis"
        if materialized is False:
            return "Validé sans modification"
        return "Validé"

    labels = {
        "passed": "Validé",
        "review_required": "À valider",
        "blocked": "Bloqué",
        "not_started": "À faire",
        "pending": "En attente",
    }

    return labels.get(
        normalized_status,
        clean_text(status).capitalize(),
    )


def build_preparation_context(
    workflow_id: str,
) -> dict[
    str,
    Any,
]:
    """
    Read only server-owned Preparation state.

    v0.2 reports certified readiness, final datasets and stage
    statuses. It does not invent detailed quality-issue counts
    because the current report-selection store does not persist
    a canonical DataQualityReport snapshot.
    """
    context: dict[
        str,
        Any,
    ] = {
        "workflow_id":
            workflow_id,

        "available":
            False,

        "ready_for_analysis":
            None,

        "session_revision":
            None,

        "dataset_count":
            None,

        "total_rows":
            None,

        "datasets":
            [],

        "stages":
            [],
    }


    try:
        from app.preparation.analysis_input_handoff import (
            load_validated_analysis_input,
        )

        from app.preparation.preparation_artifact_store import (
            list_preparation_artifacts,
        )

        from app.preparation.preparation_session import (
            get_preparation_session,
        )


        session = (
            get_preparation_session(
                workflow_id
            )
        )


        handoff = (
            load_validated_analysis_input(
                workflow_id=
                    workflow_id
            )
        )


        snapshot = getattr(
            session,
            "snapshot",
            None,
        )


        artifacts = (
            list_preparation_artifacts(
                workflow_id=
                    workflow_id
            )
        )


        materialized_stages = {
            clean_text(
                getattr(
                    artifact,
                    "stage",
                    "",
                )
            )
            .lower()

            for artifact
            in artifacts
        }


        context[
            "available"
        ] = True

        context[
            "ready_for_analysis"
        ] = bool(
            getattr(
                snapshot,
                "ready_for_analysis",
                False,
            )
        )

        context[
            "session_revision"
        ] = getattr(
            session,
            "revision",
            None,
        )

        context[
            "dataset_count"
        ] = getattr(
            handoff.ingestion,
            "dataset_count",
            None,
        )

        context[
            "total_rows"
        ] = getattr(
            handoff.ingestion,
            "total_rows",
            None,
        )


        dataset_rows: list[
            dict[
                str,
                Any,
            ]
        ] = []


        for record in (
            handoff.dataset_records
        ):
            dataframe = (
                record.get(
                    "dataframe"
                )
            )


            column_count = None


            if (
                dataframe is not None
                and
                hasattr(
                    dataframe,
                    "shape",
                )
            ):
                try:
                    column_count = int(
                        dataframe.shape[
                            1
                        ]
                    )
                except Exception:
                    column_count = None


            dataset_rows.append(
                {
                    "dataset_id":
                        clean_text(
                            record.get(
                                "dataset_id"
                            )
                        ),

                    "filename":
                        clean_text(
                            record.get(
                                "filename"
                            )
                        ),

                    "stage":
                        clean_text(
                            record.get(
                                "preparation_stage"
                            )
                        ),

                    "column_count":
                        column_count,
                }
            )


        context[
            "datasets"
        ] = dataset_rows


        stage_rows: list[
            dict[
                str,
                Any,
            ]
        ] = []


        for stage_record in (
            getattr(
                snapshot,
                "stages",
                [],
            )
            or
            []
        ):
            raw_stage = getattr(
                stage_record,
                "stage",
                "",
            )

            raw_status = getattr(
                stage_record,
                "status",
                "",
            )


            stage_name = (
                clean_text(
                    getattr(
                        raw_stage,
                        "value",
                        raw_stage,
                    )
                )
            )


            stage_rows.append(
                {
                    "stage":
                        stage_name,

                    "status":
                        clean_text(
                            getattr(
                                raw_status,
                                "value",
                                raw_status,
                            )
                        ),

                    "required":
                        bool(
                            getattr(
                                stage_record,
                                "required",
                                False,
                            )
                        ),

                    "materialized":
                        (
                            stage_name
                            .lower()
                            in
                            materialized_stages
                        ),
                }
            )


        context[
            "stages"
        ] = stage_rows


    except Exception:
        # The persisted report selection remains exportable even if
        # optional Preparation audit metadata is temporarily missing.
        pass


    return context


# ============================================================
# PAYLOAD EXTRACTION
# ============================================================

def first_executed_result(
    pipeline_payload: dict[
        str,
        Any,
    ],
) -> tuple[
    dict[
        str,
        Any,
    ]
    | None,
    dict[
        str,
        Any,
    ]
    | None,
]:
    items = (
        pipeline_payload.get(
            "items",
            [],
        )
    )


    if not isinstance(
        items,
        list,
    ):
        return (
            None,
            None,
        )


    for item in (
        items
    ):
        if not isinstance(
            item,
            dict,
        ):
            continue


        if (
            item.get(
                "pipeline_status"
            )
            !=
            "executed"
        ):
            continue


        native_tool = (
            item.get(
                "native_tool"
            )
        )


        if not isinstance(
            native_tool,
            dict,
        ):
            continue


        execution = (
            native_tool.get(
                "execution"
            )
        )


        if not isinstance(
            execution,
            dict,
        ):
            continue


        result = (
            execution.get(
                "result"
            )
        )


        if not isinstance(
            result,
            dict,
        ):
            continue


        return (
            result,
            native_tool,
        )


    return (
        None,
        None,
    )


def planner_contract_for_result(
    pipeline_payload: dict[
        str,
        Any,
    ],
) -> (
    dict[
        str,
        Any,
    ]
    | None
):
    planner = (
        pipeline_payload.get(
            "planner"
        )
    )


    if not isinstance(
        planner,
        dict,
    ):
        return None


    items = planner.get(
        "items"
    )


    if not isinstance(
        items,
        list,
    ):
        return None


    for item in (
        items
    ):
        if not isinstance(
            item,
            dict,
        ):
            continue


        if (
            item.get(
                "validation_status"
            )
            !=
            "validated"
        ):
            continue


        contract = (
            item.get(
                "contract"
            )
        )


        if isinstance(
            contract,
            dict,
        ):
            return contract


    return None


def analysis_family(
    pipeline_payload: dict[
        str,
        Any,
    ],
    result: dict[
        str,
        Any,
    ],
) -> str:
    contract = (
        planner_contract_for_result(
            pipeline_payload
        )
    )


    if isinstance(
        contract,
        dict,
    ):
        family = clean_text(
            contract.get(
                "family"
            )
        )


        if (
            family
        ):
            return family


    return clean_text(
        result.get(
            "family"
        )
    )


def ranking_rows(
    result: dict[
        str,
        Any,
    ],
) -> list[
    tuple[
        str,
        float,
    ]
]:
    data = (
        result.get(
            "chart_data",
            [],
        )
    )


    if not isinstance(
        data,
        list,
    ):
        return []


    rows: list[
        tuple[
            str,
            float,
        ]
    ] = []


    for datum in (
        data
    ):
        if not isinstance(
            datum,
            dict,
        ):
            continue


        label = clean_text(
            datum.get(
                "category"
            )
            or
            datum.get(
                "group"
            )
            or
            datum.get(
                "label"
            )
        )


        value = (
            safe_number(
                datum.get(
                    "value"
                )
            )
        )


        if (
            label
            and
            value is not None
        ):
            rows.append(
                (
                    label,
                    value,
                )
            )


    return rows


def deterministic_insight(
    detail: dict[
        str,
        Any,
    ],
) -> str:
    pipeline_payload = (
        detail.get(
            "pipeline_payload"
        )
    )


    if not isinstance(
        pipeline_payload,
        dict,
    ):
        return ""


    (
        result,
        _,
    ) = (
        first_executed_result(
            pipeline_payload
        )
    )


    if (
        result is None
    ):
        return ""


    family = (
        analysis_family(
            pipeline_payload,
            result,
        )
    )


    metrics = (
        result.get(
            "metrics"
        )
    )


    if not isinstance(
        metrics,
        dict,
    ):
        metrics = {}


    if (
        family ==
        "ranking"
    ):
        rows = (
            ranking_rows(
                result
            )
        )


        if (
            rows
        ):
            label, value = (
                rows[
                    0
                ]
            )


            return (
                f"{label} arrive en tête du classement "
                f"avec une valeur de {format_number(value)}."
            )


    if (
        family ==
        "group_comparison"
    ):
        group_count = (
            metrics.get(
                "group_count"
            )
            or
            metrics.get(
                "available_group_count"
            )
        )

        observation_count = (
            metrics.get(
                "valid_observations"
            )
            or
            metrics.get(
                "source_observation_count"
            )
        )


        contract = (
            planner_contract_for_result(
                pipeline_payload
            )
        )


        value_column = ""


        if isinstance(
            contract,
            dict,
        ):
            bindings = (
                contract.get(
                    "bindings",
                    [],
                )
            )


            if isinstance(
                bindings,
                list,
            ):
                for binding in (
                    bindings
                ):
                    if (
                        isinstance(
                            binding,
                            dict,
                        )
                        and
                        binding.get(
                            "role"
                        )
                        ==
                        "value"
                    ):
                        value_column = clean_text(
                            binding.get(
                                "column"
                            )
                        )

                        break


        if (
            group_count is not None
            and
            observation_count is not None
        ):
            return (
                f"{format_number(group_count)} catégories ont été "
                f"comparées sur {value_column or 'la métrique demandée'} "
                f"à partir de {format_number(observation_count)} observations."
            )


    summary = (
        result.get(
            "summary"
        )
    )


    if isinstance(
        summary,
        list,
    ) and summary:
        return (
            normalize_business_summary(
                summary[
                    0
                ]
            )
        )


    return ""


# ============================================================
# STYLES
# ============================================================

def build_styles() -> dict[
    str,
    ParagraphStyle,
]:
    base = (
        getSampleStyleSheet()
    )


    return {
        "title":
            ParagraphStyle(
                "SelectedReportTitle",
                parent=
                    base[
                        "Title"
                    ],
                fontName=
                    "Helvetica-Bold",
                fontSize=
                    24,
                leading=
                    29,
                textColor=
                    INK,
                spaceAfter=
                    8,
                alignment=
                    TA_LEFT,
            ),

        "subtitle":
            ParagraphStyle(
                "SelectedReportSubtitle",
                parent=
                    base[
                        "BodyText"
                    ],
                fontName=
                    "Helvetica",
                fontSize=
                    10.5,
                leading=
                    15,
                textColor=
                    MUTED,
                spaceAfter=
                    10,
            ),

        "h1":
            ParagraphStyle(
                "SelectedReportH1",
                parent=
                    base[
                        "Heading1"
                    ],
                fontName=
                    "Helvetica-Bold",
                fontSize=
                    17,
                leading=
                    21,
                textColor=
                    INK,
                spaceBefore=
                    8,
                spaceAfter=
                    10,
            ),

        "h2":
            ParagraphStyle(
                "SelectedReportH2",
                parent=
                    base[
                        "Heading2"
                    ],
                fontName=
                    "Helvetica-Bold",
                fontSize=
                    12.5,
                leading=
                    16,
                textColor=
                    INK,
                spaceBefore=
                    5,
                spaceAfter=
                    6,
            ),

        "body":
            ParagraphStyle(
                "SelectedReportBody",
                parent=
                    base[
                        "BodyText"
                    ],
                fontName=
                    "Helvetica",
                fontSize=
                    9.5,
                leading=
                    13.5,
                textColor=
                    INK,
                spaceAfter=
                    5,
            ),

        "small":
            ParagraphStyle(
                "SelectedReportSmall",
                parent=
                    base[
                        "BodyText"
                    ],
                fontName=
                    "Helvetica",
                fontSize=
                    7.8,
                leading=
                    10.5,
                textColor=
                    MUTED,
                spaceAfter=
                    3,
            ),

        "label":
            ParagraphStyle(
                "SelectedReportLabel",
                parent=
                    base[
                        "BodyText"
                    ],
                fontName=
                    "Helvetica-Bold",
                fontSize=
                    7.2,
                leading=
                    9,
                textColor=
                    MUTED,
                spaceAfter=
                    2,
            ),

        "metric":
            ParagraphStyle(
                "SelectedReportMetric",
                parent=
                    base[
                        "BodyText"
                    ],
                fontName=
                    "Helvetica-Bold",
                fontSize=
                    12.5,
                leading=
                    15,
                textColor=
                    INK,
            ),

        "callout":
            ParagraphStyle(
                "SelectedReportCallout",
                parent=
                    base[
                        "BodyText"
                    ],
                fontName=
                    "Helvetica",
                fontSize=
                    9.7,
                leading=
                    14.2,
                textColor=
                    ACCENT_DARK,
                leftIndent=
                    4,
                rightIndent=
                    4,
            ),
    }


# ============================================================
# PAGE CHROME
# ============================================================

def page_decorator(
    canvas,
    document,
) -> None:
    canvas.saveState()


    width, _ = (
        A4
    )


    canvas.setStrokeColor(
        GRID
    )

    canvas.setLineWidth(
        0.5
    )

    canvas.line(
        document.leftMargin,
        13
        *
        mm,
        width
        -
        document.rightMargin,
        13
        *
        mm,
    )


    canvas.setFont(
        "Helvetica",
        7,
    )

    canvas.setFillColor(
        MUTED
    )

    canvas.drawString(
        document.leftMargin,
        8.8
        *
        mm,
        (
            "DataLens - rapport d'analyse vérifié"
        ),
    )


    canvas.drawRightString(
        width
        -
        document.rightMargin,
        8.8
        *
        mm,
        f"Page {document.page}",
    )


    canvas.restoreState()


# ============================================================
# TABLE HELPERS
# ============================================================

def metric_table(
    metrics: list[
        tuple[
            str,
            object,
        ]
    ],
    styles: dict[
        str,
        ParagraphStyle,
    ],
) -> Table:
    cells = [
        [
            Paragraph(
                clean_text(
                    label
                ),
                styles[
                    "label"
                ],
            ),
            Paragraph(
                format_number(
                    value
                ),
                styles[
                    "metric"
                ],
            ),
        ]

        for (
            label,
            value,
        )
        in metrics
    ]


    table = (
        Table(
            cells,
            colWidths=[
                42
                *
                mm,
                44
                *
                mm,
            ],
            hAlign=
                "LEFT",
        )
    )


    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    PANEL,
                ),
                (
                    "BOX",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    0.5,
                    GRID,
                ),
                (
                    "INNERGRID",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    0.25,
                    GRID,
                ),
                (
                    "VALIGN",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    "MIDDLE",
                ),
                (
                    "LEFTPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    7,
                ),
                (
                    "RIGHTPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    7,
                ),
                (
                    "TOPPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    6,
                ),
            ]
        )
    )


    return table


def source_badge(
    source_type: str,
    styles: dict[
        str,
        ParagraphStyle,
    ],
) -> Table:
    style = (
        ParagraphStyle(
            "SelectedSourceBadge",
            parent=
                styles[
                    "small"
                ],
            fontName=
                "Helvetica-Bold",
            fontSize=
                7.2,
            leading=
                9,
            textColor=
                SUCCESS,
        )
    )


    table = (
        Table(
            [
                [
                    Paragraph(
                        source_type_label(
                            source_type
                        )
                        .upper(),
                        style,
                    )
                ]
            ],
            hAlign=
                "LEFT",
        )
    )


    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    colors.HexColor(
                        "#E8F5F1"
                    ),
                ),
                (
                    "BOX",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    0.4,
                    colors.HexColor(
                        "#D4EEE5"
                    ),
                ),
                (
                    "LEFTPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    6,
                ),
                (
                    "RIGHTPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    6,
                ),
                (
                    "TOPPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    3,
                ),
                (
                    "BOTTOMPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    3,
                ),
            ]
        )
    )


    return table


# ============================================================
# CHARTS
# ============================================================

def empty_chart(
    message: str,
) -> Drawing:
    drawing = (
        Drawing(
            500,
            70,
        )
    )


    drawing.add(
        Rect(
            0,
            0,
            500,
            70,
            fillColor=
                PANEL,
            strokeColor=
                GRID,
        )
    )


    drawing.add(
        String(
            12,
            30,
            clean_text(
                message
            ),
            fontName=
                "Helvetica",
            fontSize=
                8,
            fillColor=
                MUTED,
        )
    )


    return drawing


def horizontal_bar_chart(
    data: list[
        dict[
            str,
            Any,
        ]
    ],
) -> Drawing:
    rows: list[
        tuple[
            str,
            float,
        ]
    ] = []


    for datum in (
        data
    ):
        if not isinstance(
            datum,
            dict,
        ):
            continue


        label = clean_text(
            datum.get(
                "category"
            )
            or
            datum.get(
                "group"
            )
            or
            datum.get(
                "label"
            )
            or
            datum.get(
                "period"
            )
        )


        value = (
            safe_number(
                datum.get(
                    "value"
                )
            )
        )


        if (
            value is None
        ):
            value = (
                safe_number(
                    datum.get(
                        "mean"
                    )
                )
            )


        if (
            label
            and
            value is not None
        ):
            rows.append(
                (
                    label,
                    value,
                )
            )


    if not (
        rows
    ):
        return (
            empty_chart(
                "Aucune valeur graphique exploitable."
            )
        )


    rows = rows[
        :
        10
    ]


    drawing = (
        Drawing(
            500,
            max(
                130,
                38
                *
                len(
                    rows
                )
                +
                35,
            ),
        )
    )


    left = (
        125
    )

    width = (
        335
    )

    bar_height = (
        15
    )

    row_gap = (
        34
    )

    values = [
        value

        for (
            _,
            value,
        )
        in rows
    ]


    minimum = min(
        0.0,
        min(
            values
        ),
    )

    maximum = max(
        0.0,
        max(
            values
        ),
    )

    span = (
        maximum
        -
        minimum
    ) or 1.0


    zero_x = (
        left
        +
        (
            (
                0.0
                -
                minimum
            )
            /
            span
        )
        *
        width
    )


    drawing.add(
        Line(
            zero_x,
            20,
            zero_x,
            drawing.height
            -
            15,
            strokeColor=
                GRID,
            strokeWidth=
                0.6,
        )
    )


    for (
        index,
        (
            label,
            value,
        ),
    ) in enumerate(
        rows
    ):
        y = (
            drawing.height
            -
            35
            -
            index
            *
            row_gap
        )


        value_x = (
            left
            +
            (
                (
                    value
                    -
                    minimum
                )
                /
                span
            )
            *
            width
        )


        x = min(
            zero_x,
            value_x,
        )

        bar_width = abs(
            value_x
            -
            zero_x
        )


        drawing.add(
            String(
                4,
                y
                +
                3,
                truncate(
                    label,
                    24,
                ),
                fontName=
                    "Helvetica",
                fontSize=
                    7.5,
                fillColor=
                    INK,
            )
        )


        drawing.add(
            Rect(
                x,
                y,
                max(
                    1,
                    bar_width,
                ),
                bar_height,
                fillColor=
                    ACCENT_LIGHT,
                strokeColor=
                    ACCENT,
                strokeWidth=
                    0.7,
            )
        )


        drawing.add(
            String(
                min(
                    465,
                    max(
                        130,
                        value_x
                        +
                        5,
                    ),
                ),
                y
                +
                3,
                format_number(
                    value
                ),
                fontName=
                    "Helvetica-Bold",
                fontSize=
                    7,
                fillColor=
                    INK,
            )
        )


    return drawing


def boxplot_chart(
    data: list[
        dict[
            str,
            Any,
        ]
    ],
) -> Drawing:
    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []


    values: list[
        float
    ] = []


    for datum in (
        data
    ):
        if not isinstance(
            datum,
            dict,
        ):
            continue


        required = {
            key:
                safe_number(
                    datum.get(
                        key
                    )
                )

            for key
            in (
                "min",
                "q1",
                "median",
                "q3",
                "max",
            )
        }


        if any(
            value is None

            for value
            in required.values()
        ):
            continue


        group_value = datum.get(
            "group"
        )


        if (
            group_value
            is None
        ):
            group_value = datum.get(
                "category"
            )


        label = clean_text(
            str(
                group_value
            )
        )


        if not (
            label
        ):
            continue


        row = {
            "label":
                label,

            **required,
        }


        rows.append(
            row
        )


        values.extend(
            [
                float(
                    required[
                        "min"
                    ]
                ),
                float(
                    required[
                        "max"
                    ]
                ),
            ]
        )


    if not (
        rows
    ):
        return (
            empty_chart(
                "Pas de statistiques de boîte à moustaches."
            )
        )


    rows = rows[
        :
        8
    ]


    minimum = min(
        values
    )

    maximum = max(
        values
    )

    span = (
        maximum
        -
        minimum
    ) or 1.0


    drawing = (
        Drawing(
            500,
            max(
                140,
                42
                *
                len(
                    rows
                )
                +
                40,
            ),
        )
    )


    left = (
        125
    )

    width = (
        345
    )


    def scale(
        value: float,
    ) -> float:
        return (
            left
            +
            (
                (
                    value
                    -
                    minimum
                )
                /
                span
            )
            *
            width
        )


    for (
        index,
        row,
    ) in enumerate(
        rows
    ):
        y = (
            drawing.height
            -
            40
            -
            index
            *
            42
        )


        minimum_x = scale(
            float(
                row[
                    "min"
                ]
            )
        )

        q1_x = scale(
            float(
                row[
                    "q1"
                ]
            )
        )

        median_x = scale(
            float(
                row[
                    "median"
                ]
            )
        )

        q3_x = scale(
            float(
                row[
                    "q3"
                ]
            )
        )

        maximum_x = scale(
            float(
                row[
                    "max"
                ]
            )
        )


        drawing.add(
            String(
                4,
                y
                -
                2,
                truncate(
                    row[
                        "label"
                    ],
                    24,
                ),
                fontName=
                    "Helvetica",
                fontSize=
                    7.5,
                fillColor=
                    INK,
            )
        )


        drawing.add(
            Line(
                minimum_x,
                y,
                maximum_x,
                y,
                strokeColor=
                    MUTED,
                strokeWidth=
                    0.8,
            )
        )


        drawing.add(
            Line(
                minimum_x,
                y
                -
                5,
                minimum_x,
                y
                +
                5,
                strokeColor=
                    MUTED,
            )
        )


        drawing.add(
            Line(
                maximum_x,
                y
                -
                5,
                maximum_x,
                y
                +
                5,
                strokeColor=
                    MUTED,
            )
        )


        drawing.add(
            Rect(
                q1_x,
                y
                -
                8,
                max(
                    1,
                    q3_x
                    -
                    q1_x,
                ),
                16,
                fillColor=
                    ACCENT_LIGHT,
                strokeColor=
                    ACCENT,
                strokeWidth=
                    0.8,
            )
        )


        drawing.add(
            Line(
                median_x,
                y
                -
                8,
                median_x,
                y
                +
                8,
                strokeColor=
                    INK,
                strokeWidth=
                    1.2,
            )
        )


    drawing.add(
        String(
            left,
            10,
            format_number(
                minimum
            ),
            fontSize=
                6.5,
            fillColor=
                MUTED,
        )
    )


    drawing.add(
        String(
            left
            +
            width
            -
            25,
            10,
            format_number(
                maximum
            ),
            fontSize=
                6.5,
            fillColor=
                MUTED,
        )
    )


    return drawing


def time_series_value_label_indices(
    values: list[
        float
    ],
) -> list[
    int
]:
    """
    Select a small deterministic set of points whose values
    should be printed directly on a static PDF time-series
    chart.

    The purpose is readability, not analytical sampling:
    the complete series is still drawn.

    Important points such as the first observation, last
    observation, minimum and maximum are always retained.
    """

    count = len(
        values
    )


    if count == 0:
        return []


    if count <= 8:
        selected = set(
            range(
                count
            )
        )

    elif count <= 16:
        selected = set(
            range(
                0,
                count,
                2,
            )
        )

    elif count <= 30:
        selected = set(
            range(
                0,
                count,
                4,
            )
        )

    else:
        selected = {
            0,
            count // 4,
            count // 2,
            (
                count
                *
                3
            )
            //
            4,
            count - 1,
        }


    selected.add(
        0
    )

    selected.add(
        count - 1
    )


    minimum_index = min(
        range(
            count
        ),
        key=lambda index:
            values[
                index
            ],
    )


    maximum_index = max(
        range(
            count
        ),
        key=lambda index:
            values[
                index
            ],
    )


    selected.add(
        minimum_index
    )

    selected.add(
        maximum_index
    )


    return sorted(
        selected
    )


def format_chart_value_label(
    value: object,
) -> str:
    """
    Compact number formatter dedicated to labels drawn
    directly inside charts.

    Report tables keep the full precise formatting.
    """

    number = safe_number(
        value
    )


    if number is None:
        return ""


    absolute = abs(
        number
    )


    if (
        absolute
        >=
        1_000_000
    ):
        compact = (
            f"{number / 1_000_000:.1f}"
            .replace(
                ".",
                ",",
            )
        )

        return (
            f"{compact} M"
        )


    if (
        absolute
        >=
        100_000
    ):
        compact = (
            f"{number / 1_000:.0f}"
        )

        return (
            f"{compact} k"
        )


    if (
        absolute
        >=
        1_000
    ):
        return format_number(
            number,
            decimals=0,
        )


    if float(
        number
    ).is_integer():
        return format_number(
            number,
            decimals=0,
        )


    return format_number(
        number,
        decimals=1,
    )


def time_series_line_chart(
    data: list[
        dict[
            str,
            Any,
        ]
    ],
) -> Drawing:
    points: list[
        tuple[
            int,
            float,
            dict[
                str,
                Any,
            ],
        ]
    ] = []


    for index, datum in enumerate(
        data
    ):
        if not isinstance(
            datum,
            dict,
        ):
            continue


        value = safe_number(
            datum.get(
                "value"
            )
        )


        if value is None:
            value = safe_number(
                datum.get(
                    "median"
                )
            )


        if value is not None:
            points.append(
                (
                    index,
                    value,
                    datum,
                )
            )


    if len(
        points
    ) < 2:
        return empty_chart(
            "Pas assez de points pour la s\u00e9rie temporelle."
        )


    drawing = Drawing(
        500,
        215,
    )

    left = 52
    bottom = 35
    width = 425
    height = 150


    values = [
        value
        for _, value, _ in points
    ]


    minimum = min(
        values
    )

    maximum = max(
        values
    )

    span = (
        maximum
        -
        minimum
    ) or 1.0


    # ========================================================
    # AXES
    # ========================================================

    drawing.add(
        Line(
            left,
            bottom,
            left,
            bottom + height,
            strokeColor=GRID,
            strokeWidth=0.7,
        )
    )


    drawing.add(
        Line(
            left,
            bottom,
            left + width,
            bottom,
            strokeColor=GRID,
            strokeWidth=0.7,
        )
    )


    for ratio in (
        0.0,
        0.5,
        1.0,
    ):
        y = (
            bottom
            +
            ratio
            *
            height
        )


        drawing.add(
            Line(
                left,
                y,
                left + width,
                y,
                strokeColor=GRID,
                strokeWidth=0.35,
            )
        )


        axis_value = (
            minimum
            +
            ratio
            *
            span
        )


        drawing.add(
            String(
                2,
                y - 2,
                format_number(
                    axis_value
                ),
                fontName="Helvetica",
                fontSize=6.5,
                fillColor=MUTED,
            )
        )


    # ========================================================
    # MAIN SERIES
    # ========================================================

    projected: list[
        float
    ] = []


    projected_points: list[
        tuple[
            float,
            float,
            float,
        ]
    ] = []


    for offset, (
        _,
        value,
        _,
    ) in enumerate(
        points
    ):
        x = (
            left
            +
            (
                offset
                /
                max(
                    len(points) - 1,
                    1,
                )
            )
            *
            width
        )


        y = (
            bottom
            +
            (
                (
                    value
                    -
                    minimum
                )
                /
                span
            )
            *
            height
        )


        projected.extend(
            [
                x,
                y,
            ]
        )


        projected_points.append(
            (
                x,
                y,
                value,
            )
        )


    drawing.add(
        PolyLine(
            projected,
            strokeColor=ACCENT,
            strokeWidth=2.0,
            fillColor=None,
        )
    )


    # ========================================================
    # MOVING AVERAGE
    # ========================================================

    moving: list[
        float
    ] = []


    for offset, (
        _,
        _,
        datum,
    ) in enumerate(
        points
    ):
        moving_value = safe_number(
            datum.get(
                "moving_average"
            )
        )


        if moving_value is None:
            continue


        x = (
            left
            +
            (
                offset
                /
                max(
                    len(points) - 1,
                    1,
                )
            )
            *
            width
        )


        y = (
            bottom
            +
            (
                (
                    moving_value
                    -
                    minimum
                )
                /
                span
            )
            *
            height
        )


        moving.extend(
            [
                x,
                y,
            ]
        )


    if len(
        moving
    ) >= 4:
        drawing.add(
            PolyLine(
                moving,
                strokeColor=MUTED,
                strokeWidth=1.4,
                fillColor=None,
                strokeDashArray=[
                    5,
                    4,
                ],
            )
        )


    # ========================================================
    # VALUE LABELS
    #
    # A PDF has no hover tooltip. Selected values are printed
    # directly on the chart while the full series remains
    # visible.
    # ========================================================

    label_indices = (
        time_series_value_label_indices(
            values
        )
    )


    for label_index in (
        label_indices
    ):
        (
            x,
            y,
            value,
        ) = (
            projected_points[
                label_index
            ]
        )


        label = (
            format_chart_value_label(
                value
            )
        )


        if not label:
            continue


        # Small visual anchor on the line.
        drawing.add(
            Rect(
                x - 1.5,
                y - 1.5,
                3,
                3,
                fillColor=ACCENT,
                strokeColor=ACCENT,
                strokeWidth=0,
            )
        )


        # Labels close to the upper edge are placed below
        # their point so they remain inside the drawing.
        if (
            y
            >
            bottom
            +
            height
            -
            18
        ):
            label_y = (
                y
                -
                11
            )

        else:
            label_y = (
                y
                +
                6
            )


        estimated_half_width = (
            len(
                label
            )
            *
            1.45
        )


        label_x = (
            x
            -
            estimated_half_width
        )


        label_x = max(
            left,
            min(
                label_x,
                left
                +
                width
                -
                (
                    len(
                        label
                    )
                    *
                    2.9
                ),
            ),
        )


        drawing.add(
            String(
                label_x,
                label_y,
                label,
                fontName="Helvetica-Bold",
                fontSize=5.8,
                fillColor=INK,
            )
        )


    # ========================================================
    # PERIOD LABELS
    # ========================================================

    def period_label(
        datum: dict[
            str,
            Any,
        ],
    ) -> str:
        raw = (
            datum.get(
                "period"
            )
            or
            datum.get(
                "time"
            )
        )


        if raw is None:
            return ""


        raw_text = clean_text(
            raw
        )


        try:
            parsed = datetime.fromisoformat(
                raw_text.replace(
                    "Z",
                    "+00:00",
                )
            )


            return parsed.strftime(
                "%m/%Y"
            )


        except Exception:
            return truncate(
                raw_text,
                14,
            )


    start_label = (
        period_label(
            points[
                0
            ][
                2
            ]
        )
        or
        "D\u00e9but"
    )


    end_label = (
        period_label(
            points[
                -1
            ][
                2
            ]
        )
        or
        "Fin"
    )


    drawing.add(
        String(
            left,
            12,
            start_label,
            fontName="Helvetica",
            fontSize=6.5,
            fillColor=MUTED,
        )
    )


    drawing.add(
        String(
            left + width - 35,
            12,
            end_label,
            fontName="Helvetica",
            fontSize=6.5,
            fillColor=MUTED,
        )
    )


    drawing.add(
        String(
            left,
            drawing.height - 12,
            "\u00c9volution par p\u00e9riode",
            fontName="Helvetica-Bold",
            fontSize=7.3,
            fillColor=INK,
        )
    )


    return drawing


def pdf_variable_label(
    value: object,
) -> str:
    """
    Presentation-only label for common analytical variables.

    Backend column names and analytical contracts remain unchanged.
    """

    normalized = clean_text(
        value
    )


    labels = {
        "age_at_first_purchase":
            "\u00c2ge au premier achat",

        "total_spend":
            "Montant total",

        "purchase_sessions":
            "Fr\u00e9quence d'achat",

        "average_basket":
            "Panier moyen",

        "gender":
            "Genre",

        "category":
            "Cat\u00e9gorie",

        "categ":
            "Cat\u00e9gorie",

        "client_id":
            "Client",
    }


    if normalized in labels:
        return labels[
            normalized
        ]


    fallback = (
        normalized
        .replace(
            "_",
            " ",
        )
        .strip()
    )


    if not fallback:
        return "Variable"


    return (
        fallback[
            0
        ].upper()
        +
        fallback[
            1:
        ]
    )


def bounded_pdf_scatter_data(
    data: list[
        dict[
            str,
            Any,
        ]
    ],
    *,
    max_points: int = 350,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """
    Bound only the static PDF visual payload.

    Analytical calculations and persisted statistics remain based
    on the complete analytical dataset.
    """

    valid: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for datum in data:
        if not isinstance(
            datum,
            dict,
        ):
            continue


        x_value = safe_number(
            datum.get(
                "x"
            )
        )

        y_value = safe_number(
            datum.get(
                "y"
            )
        )


        if (
            x_value is None
            or
            y_value is None
        ):
            continue


        valid.append(
            datum
        )


    if (
        max_points <= 0
        or
        len(
            valid
        )
        <=
        max_points
    ):
        return valid


    if max_points == 1:
        return [
            valid[
                0
            ]
        ]


    last_index = (
        len(
            valid
        )
        -
        1
    )


    selected_indices = sorted(
        {
            round(
                position
                *
                last_index
                /
                (
                    max_points
                    -
                    1
                )
            )

            for position
            in range(
                max_points
            )
        }
    )


    return [
        valid[
            index
        ]

        for index
        in selected_indices
    ]


def scatter_chart(
    data: list[
        dict[
            str,
            Any,
        ]
    ],
    *,
    x_label: object = "",
    y_label: object = "",
) -> Drawing:
    valid_data = (
        bounded_pdf_scatter_data(
            data
        )
    )


    if len(
        valid_data
    ) < 2:
        return empty_chart(
            "Pas assez de points pour le nuage de points."
        )


    points: list[
        tuple[
            float,
            float,
        ]
    ] = []


    for datum in valid_data:
        x_value = safe_number(
            datum.get(
                "x"
            )
        )

        y_value = safe_number(
            datum.get(
                "y"
            )
        )


        if (
            x_value is None
            or
            y_value is None
        ):
            continue


        points.append(
            (
                x_value,
                y_value,
            )
        )


    if len(
        points
    ) < 2:
        return empty_chart(
            "Pas assez de points pour le nuage de points."
        )


    drawing = Drawing(
        500,
        215,
    )


    left = 58
    bottom = 40
    width = 415
    height = 140


    x_values = [
        x
        for x, _
        in points
    ]

    y_values = [
        y
        for _, y
        in points
    ]


    x_minimum = min(
        x_values
    )

    x_maximum = max(
        x_values
    )

    y_minimum = min(
        y_values
    )

    y_maximum = max(
        y_values
    )


    x_span = (
        x_maximum
        -
        x_minimum
    ) or 1.0

    y_span = (
        y_maximum
        -
        y_minimum
    ) or 1.0


    # ========================================================
    # GRID
    # ========================================================

    for ratio in (
        0.0,
        0.5,
        1.0,
    ):
        x = (
            left
            +
            ratio
            *
            width
        )

        y = (
            bottom
            +
            ratio
            *
            height
        )


        drawing.add(
            Line(
                x,
                bottom,
                x,
                bottom + height,
                strokeColor=GRID,
                strokeWidth=0.35,
            )
        )


        drawing.add(
            Line(
                left,
                y,
                left + width,
                y,
                strokeColor=GRID,
                strokeWidth=0.35,
            )
        )


        y_axis_value = (
            y_minimum
            +
            ratio
            *
            y_span
        )


        drawing.add(
            String(
                2,
                y - 2,
                format_chart_value_label(
                    y_axis_value
                ),
                fontName="Helvetica",
                fontSize=6.2,
                fillColor=MUTED,
            )
        )


    # ========================================================
    # POINTS
    # ========================================================

    for (
        x_value,
        y_value,
    ) in points:
        x = (
            left
            +
            (
                (
                    x_value
                    -
                    x_minimum
                )
                /
                x_span
            )
            *
            width
        )

        y = (
            bottom
            +
            (
                (
                    y_value
                    -
                    y_minimum
                )
                /
                y_span
            )
            *
            height
        )


        drawing.add(
            Rect(
                x - 0.8,
                y - 0.8,
                1.6,
                1.6,
                fillColor=ACCENT,
                strokeColor=None,
            )
        )


    # ========================================================
    # AXIS VALUES
    # ========================================================

    drawing.add(
        String(
            left,
            25,
            format_chart_value_label(
                x_minimum
            ),
            fontName="Helvetica",
            fontSize=6.2,
            fillColor=MUTED,
        )
    )


    maximum_label = (
        format_chart_value_label(
            x_maximum
        )
    )


    drawing.add(
        String(
            left
            +
            width
            -
            (
                len(
                    maximum_label
                )
                *
                3.0
            ),
            25,
            maximum_label,
            fontName="Helvetica",
            fontSize=6.2,
            fillColor=MUTED,
        )
    )


    # ========================================================
    # PRESENTATION LABELS
    # ========================================================

    rendered_x_label = (
        pdf_variable_label(
            x_label
        )
    )

    rendered_y_label = (
        pdf_variable_label(
            y_label
        )
    )


    drawing.add(
        String(
            left,
            drawing.height - 12,
            (
                f"{rendered_y_label} selon "
                f"{rendered_x_label}"
            ),
            fontName="Helvetica-Bold",
            fontSize=7.3,
            fillColor=INK,
        )
    )


    drawing.add(
        String(
            left,
            10,
            (
                f"X : "
                f"{rendered_x_label}"
            ),
            fontName="Helvetica",
            fontSize=6.3,
            fillColor=MUTED,
        )
    )


    drawing.add(
        String(
            left + 180,
            10,
            (
                f"Y : "
                f"{rendered_y_label}"
            ),
            fontName="Helvetica",
            fontSize=6.3,
            fillColor=MUTED,
        )
    )


    if (
        len(
            valid_data
        )
        <
        len(
            data
        )
    ):
        drawing.add(
            String(
                350,
                drawing.height - 12,
                (
                    f"{len(valid_data)} / "
                    f"{len(data)} points affich\u00e9s"
                ),
                fontName="Helvetica",
                fontSize=5.8,
                fillColor=MUTED,
            )
        )


    return drawing


def heatmap_chart(
    data: list[
        dict[
            str,
            Any,
        ]
    ],
    *,
    x_label: object = "",
    y_label: object = "",
) -> Drawing:
    cells: list[
        tuple[
            str,
            str,
            float,
        ]
    ] = []


    for datum in data:
        if not isinstance(
            datum,
            dict,
        ):
            continue


        count = safe_number(
            datum.get(
                "count"
            )
        )


        if count is None:
            continue


        x_value = clean_text(
            datum.get(
                "x"
            )
        )

        y_value = clean_text(
            datum.get(
                "y"
            )
        )


        if (
            not x_value
            or
            not y_value
        ):
            continue


        cells.append(
            (
                x_value,
                y_value,
                count,
            )
        )


    if not cells:
        return empty_chart(
            "Aucune cellule disponible pour la table crois\u00e9e."
        )


    x_levels: list[
        str
    ] = []

    y_levels: list[
        str
    ] = []


    for (
        x_value,
        y_value,
        _,
    ) in cells:
        if (
            x_value
            not in
            x_levels
        ):
            x_levels.append(
                x_value
            )


        if (
            y_value
            not in
            y_levels
        ):
            y_levels.append(
                y_value
            )


    if (
        not x_levels
        or
        not y_levels
    ):
        return empty_chart(
            "Aucune modalit\u00e9 disponible pour la table crois\u00e9e."
        )


    drawing = Drawing(
        500,
        215,
    )


    left = 72
    bottom = 38
    width = 400
    height = 130


    cell_width = (
        width
        /
        len(
            x_levels
        )
    )

    cell_height = (
        height
        /
        len(
            y_levels
        )
    )


    maximum_count = max(
        count
        for _, _, count
        in cells
    ) or 1.0


    cell_lookup = {
        (
            x_value,
            y_value,
        ):
            count

        for (
            x_value,
            y_value,
            count,
        )
        in cells
    }


    # ========================================================
    # CELLS
    # ========================================================

    for y_index, y_value in enumerate(
        y_levels
    ):
        for x_index, x_value in enumerate(
            x_levels
        ):
            count = (
                cell_lookup.get(
                    (
                        x_value,
                        y_value,
                    ),
                    0.0,
                )
            )


            ratio = (
                count
                /
                maximum_count
            )


            if ratio >= 0.67:
                fill_color = ACCENT
                text_color = colors.white

            elif ratio >= 0.34:
                fill_color = PANEL_BLUE
                text_color = INK

            else:
                fill_color = PANEL
                text_color = INK


            x = (
                left
                +
                x_index
                *
                cell_width
            )

            y = (
                bottom
                +
                (
                    len(
                        y_levels
                    )
                    -
                    y_index
                    -
                    1
                )
                *
                cell_height
            )


            drawing.add(
                Rect(
                    x,
                    y,
                    cell_width,
                    cell_height,
                    fillColor=fill_color,
                    strokeColor=GRID,
                    strokeWidth=0.5,
                )
            )


            count_label = (
                format_number(
                    count,
                    decimals=0,
                )
            )


            drawing.add(
                String(
                    x
                    +
                    (
                        cell_width
                        /
                        2
                    )
                    -
                    (
                        len(
                            count_label
                        )
                        *
                        1.7
                    ),
                    y
                    +
                    (
                        cell_height
                        /
                        2
                    )
                    -
                    2,
                    count_label,
                    fontName="Helvetica-Bold",
                    fontSize=7.0,
                    fillColor=text_color,
                )
            )


    # ========================================================
    # LEVEL LABELS
    # ========================================================

    for x_index, x_value in enumerate(
        x_levels
    ):
        drawing.add(
            String(
                left
                +
                x_index
                *
                cell_width
                +
                (
                    cell_width
                    /
                    2
                )
                -
                (
                    len(
                        x_value
                    )
                    *
                    1.5
                ),
                24,
                x_value,
                fontName="Helvetica-Bold",
                fontSize=6.5,
                fillColor=INK,
            )
        )


    for y_index, y_value in enumerate(
        y_levels
    ):
        y = (
            bottom
            +
            (
                len(
                    y_levels
                )
                -
                y_index
                -
                1
            )
            *
            cell_height
            +
            (
                cell_height
                /
                2
            )
            -
            2
        )


        drawing.add(
            String(
                42,
                y,
                y_value,
                fontName="Helvetica-Bold",
                fontSize=6.5,
                fillColor=INK,
            )
        )


    rendered_x_label = (
        pdf_variable_label(
            x_label
        )
    )

    rendered_y_label = (
        pdf_variable_label(
            y_label
        )
    )


    drawing.add(
        String(
            left,
            drawing.height - 12,
            "Table crois\u00e9e des effectifs",
            fontName="Helvetica-Bold",
            fontSize=7.3,
            fillColor=INK,
        )
    )


    drawing.add(
        String(
            left,
            10,
            (
                f"Colonnes : "
                f"{rendered_x_label}"
            ),
            fontName="Helvetica",
            fontSize=6.2,
            fillColor=MUTED,
        )
    )


    drawing.add(
        String(
            left + 180,
            10,
            (
                f"Lignes : "
                f"{rendered_y_label}"
            ),
            fontName="Helvetica",
            fontSize=6.2,
            fillColor=MUTED,
        )
    )


    return drawing



def lorenz_chart(
    data: list[
        dict[
            str,
            Any,
        ]
    ],
) -> Drawing:
    """
    Render a Lorenz curve from server-owned chart data.

    No analytical quantity is recalculated here. The renderer
    consumes population_share, revenue_share and equality_share
    produced by the deterministic inequality executor.
    """

    points: list[
        tuple[
            float,
            float,
            float,
        ]
    ] = []


    for datum in data:
        if not isinstance(
            datum,
            dict,
        ):
            continue


        try:
            population_share = float(
                datum.get(
                    "population_share"
                )
            )

            revenue_share = float(
                datum.get(
                    "revenue_share"
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue


        equality_raw = datum.get(
            "equality_share"
        )


        if equality_raw is None:
            equality_share = (
                population_share
            )
        else:
            try:
                equality_share = float(
                    equality_raw
                )
            except (
                TypeError,
                ValueError,
            ):
                equality_share = (
                    population_share
                )


        points.append(
            (
                population_share,
                revenue_share,
                equality_share,
            )
        )


    if len(points) < 2:
        return empty_chart(
            "Pas assez de points pour la courbe de Lorenz."
        )


    drawing = Drawing(
        500,
        235,
    )


    left = 58
    bottom = 42
    plot_size = 160


    def project_x(
        share: float,
    ) -> float:
        bounded = max(
            0.0,
            min(
                1.0,
                share,
            ),
        )

        return (
            left +
            bounded *
            plot_size
        )


    def project_y(
        share: float,
    ) -> float:
        bounded = max(
            0.0,
            min(
                1.0,
                share,
            ),
        )

        return (
            bottom +
            bounded *
            plot_size
        )


    ticks = (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    )


    for ratio in ticks:
        x = project_x(
            ratio
        )

        y = project_y(
            ratio
        )


        drawing.add(
            Line(
                left,
                y,
                left +
                plot_size,
                y,
                strokeColor=GRID,
                strokeWidth=0.4,
            )
        )

        drawing.add(
            Line(
                x,
                bottom,
                x,
                bottom +
                plot_size,
                strokeColor=GRID,
                strokeWidth=0.4,
            )
        )


        label = (
            f"{ratio * 100:.0f}%"
        )


        drawing.add(
            String(
                x,
                26,
                label,
                textAnchor="middle",
                fontSize=6.5,
                fillColor=MUTED,
            )
        )

        drawing.add(
            String(
                left - 10,
                y - 2,
                label,
                textAnchor="end",
                fontSize=6.5,
                fillColor=MUTED,
            )
        )


    drawing.add(
        Line(
            left,
            bottom,
            left +
            plot_size,
            bottom,
            strokeColor=INK,
            strokeWidth=0.8,
        )
    )

    drawing.add(
        Line(
            left,
            bottom,
            left,
            bottom +
            plot_size,
            strokeColor=INK,
            strokeWidth=0.8,
        )
    )


    lorenz_points: list[
        float
    ] = []

    equality_points: list[
        float
    ] = []


    for (
        population_share,
        revenue_share,
        equality_share,
    ) in points:

        lorenz_points.extend(
            [
                project_x(
                    population_share
                ),
                project_y(
                    revenue_share
                ),
            ]
        )

        equality_points.extend(
            [
                project_x(
                    population_share
                ),
                project_y(
                    equality_share
                ),
            ]
        )


    drawing.add(
        PolyLine(
            equality_points,
            strokeColor=MUTED,
            strokeWidth=1.0,
            strokeDashArray=[
                5,
                4,
            ],
            fillColor=None,
        )
    )


    drawing.add(
        PolyLine(
            lorenz_points,
            strokeColor=ACCENT,
            strokeWidth=2.2,
            fillColor=None,
        )
    )


    drawing.add(
        String(
            260,
            178,
            "Courbe de Lorenz",
            fontName="Helvetica-Bold",
            fontSize=8.5,
            fillColor=INK,
        )
    )

    drawing.add(
        Line(
            260,
            166,
            300,
            166,
            strokeColor=ACCENT,
            strokeWidth=2.2,
        )
    )

    drawing.add(
        String(
            308,
            163,
            "Part cumul\u00e9e observ\u00e9e",
            fontSize=7.5,
            fillColor=INK,
        )
    )


    drawing.add(
        Line(
            260,
            139,
            300,
            139,
            strokeColor=MUTED,
            strokeWidth=1.0,
            strokeDashArray=[
                5,
                4,
            ],
        )
    )

    drawing.add(
        String(
            308,
            136,
            "\u00c9galit\u00e9 parfaite",
            fontSize=7.5,
            fillColor=MUTED,
        )
    )


    drawing.add(
        String(
            260,
            93,
            "X : part cumul\u00e9e des clients",
            fontSize=7.5,
            fillColor=MUTED,
        )
    )

    drawing.add(
        String(
            260,
            76,
            "Y : part cumul\u00e9e du chiffre d'affaires",
            fontSize=7.5,
            fillColor=MUTED,
        )
    )


    drawing.add(
        String(
            left +
            plot_size / 2,
            8,
            "Part cumul\u00e9e des clients",
            textAnchor="middle",
            fontSize=7,
            fillColor=MUTED,
        )
    )


    return drawing



def result_chart(
    result: dict[
        str,
        Any,
    ],
) -> (
    Drawing
    | None
):
    data = result.get(
        "chart_data",
        [],
    )


    if not isinstance(
        data,
        list,
    ):
        return None


    if not (
        data
    ):
        return None


    chart_type = clean_text(
        result.get(
            "chart_type"
        )
    ).lower()


    metrics = result.get(
        "metrics",
        {},
    )


    if not isinstance(
        metrics,
        dict,
    ):
        metrics = {}


    if (
        chart_type
        in {
            "line",
            "line_band",
        }
    ):
        return (
            time_series_line_chart(
                data
            )
        )


    if (
        chart_type
        in {
            "bar",
            "grouped_summary",
        }
    ):
        return (
            horizontal_bar_chart(
                data
            )
        )


    if (
        chart_type ==
        "scatter"
    ):
        return (
            scatter_chart(
                data,
                x_label=metrics.get(
                    "x_column",
                    "",
                ),
                y_label=metrics.get(
                    "y_column",
                    "",
                ),
            )
        )


    if (
        chart_type ==
        "heatmap"
    ):
        return (
            heatmap_chart(
                data,
                x_label=metrics.get(
                    "x_column",
                    "",
                ),
                y_label=metrics.get(
                    "y_column",
                    "",
                ),
            )
        )


    if (
        chart_type ==
        "boxplot"
    ):
        return (
            boxplot_chart(
                data
            )
        )


    if (
        chart_type ==
        "lorenz"
    ):
        return (
            lorenz_chart(
                data
            )
        )


    return None


# ============================================================
# ANALYSIS BLOCK
# ============================================================

def group_comparison_summary_table(
    *,
    result: dict[
        str,
        Any,
    ],
    styles: dict[
        str,
        ParagraphStyle,
    ],
) -> (
    Table
    | None
):
    """
    Build a presentation-only descriptive summary of groups
    already computed by the deterministic engine.

    No inferential calculation is performed here.
    """

    data = result.get(
        "chart_data",
        [],
    )


    if not isinstance(
        data,
        list,
    ):
        return None


    metrics = result.get(
        "metrics",
        {},
    )


    if not isinstance(
        metrics,
        dict,
    ):
        metrics = {}


    group_column = clean_text(
        metrics.get(
            "group_column"
        )
    )


    value_column = clean_text(
        metrics.get(
            "value_column"
        )
    )


    group_header = (
        pdf_variable_label(
            group_column
        )
        if group_column
        else
        "Groupe"
    )


    normalized_value_column = (
        value_column
        .lower()
        .replace(
            "_",
            " ",
        )
    )


    value_suffix = (
        " ans"
        if (
            "age"
            in normalized_value_column.split()
        )
        else
        ""
    )


    rows: list[
        list[
            str
        ]
    ] = []


    for datum in data:
        if not isinstance(
            datum,
            dict,
        ):
            continue


        group_value = datum.get(
            "group"
        )

        count = safe_number(
            datum.get(
                "count"
            )
        )

        median = safe_number(
            datum.get(
                "median"
            )
        )

        q1 = safe_number(
            datum.get(
                "q1"
            )
        )

        q3 = safe_number(
            datum.get(
                "q3"
            )
        )


        if (
            group_value is None
            or
            count is None
            or
            median is None
            or
            q1 is None
            or
            q3 is None
        ):
            continue


        rows.append(
            [
                clean_text(
                    str(
                        group_value
                    )
                ),

                format_number(
                    count,
                    decimals=0,
                ),

                (
                    f"{format_number(median)}"
                    f"{value_suffix}"
                ),

                (
                    f"{format_number(q1)}"
                    f"\u2013"
                    f"{format_number(q3)}"
                    f"{value_suffix}"
                ),
            ]
        )


    if not rows:
        return None


    maximum_rows = 8


    visible_rows = rows[
        :
        maximum_rows
    ]


    table_data = [
        [
            group_header,
            "Effectif",
            "M\u00e9diane",
            "Q1\u2013Q3",
        ],
        *visible_rows,
    ]


    hidden_count = (
        len(
            rows
        )
        -
        len(
            visible_rows
        )
    )


    if (
        hidden_count
        >
        0
    ):
        table_data.append(
            [
                (
                    f"+ {hidden_count} "
                    f"autre(s) groupe(s)"
                ),
                "",
                "",
                "",
            ]
        )


    table = Table(
        table_data,
        colWidths=[
            46 * mm,
            30 * mm,
            38 * mm,
            46 * mm,
        ],
        repeatRows=1,
    )


    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        0,
                    ),
                    PANEL_BLUE,
                ),
                (
                    "FONTNAME",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        0,
                    ),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    7.0,
                ),
                (
                    "ALIGN",
                    (
                        1,
                        1,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    "RIGHT",
                ),
                (
                    "VALIGN",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    "MIDDLE",
                ),
                (
                    "BOX",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    0.45,
                    GRID,
                ),
                (
                    "INNERGRID",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    0.25,
                    GRID,
                ),
                (
                    "LEFTPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    6,
                ),
                (
                    "RIGHTPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    6,
                ),
                (
                    "TOPPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    5,
                ),
            ]
        )
    )


    return table


def time_series_period_label(
    value: object,
) -> str:
    normalized = (
        clean_text(
            value
        )
        .strip()
        .lower()
    )


    labels = {
        "day":
            "Jour",

        "week":
            "Semaine",

        "month":
            "Mois",

        "quarter":
            "Trimestre",

        "year":
            "Ann\u00e9e",
    }


    return (
        labels.get(
            normalized,
            clean_text(
                value
            ),
        )
    )


def selected_metrics(
    *,
    family: str,
    result: dict[
        str,
        Any,
    ],
) -> list[
    tuple[
        str,
        object,
    ]
]:
    metrics = (
        result.get(
            "metrics"
        )
    )


    if not isinstance(
        metrics,
        dict,
    ):
        return []


    if (
        family ==
        "descriptive_metric"
    ):
        if (
            metrics.get(
                "transaction_count"
            )
            is not None
        ):
            return [
                (
                    "Transactions",
                    metrics.get(
                        "transaction_count"
                    ),
                )
            ]


        if (
            metrics.get(
                "products_sold_count"
            )
            is not None
        ):
            output = [
                (
                    "Occurrences produit",
                    metrics.get(
                        "products_sold_count"
                    ),
                )
            ]


            if (
                metrics.get(
                    "distinct_products_sold"
                )
                is not None
            ):
                output.append(
                    (
                        "Références distinctes",
                        metrics.get(
                            "distinct_products_sold"
                        ),
                    )
                )


            return output


    if (
        family ==
        "time_series"
    ):
        result_kind = (
            clean_text(
                result.get(
                    "kind"
                )
            )
            .strip()
            .lower()
        )


        if (
            result_kind
            ==
            "revenue_moving_average"
        ):
            aggregation_period = (
                metrics.get(
                    "aggregation_period"
                )
            )

            moving_average_window = (
                metrics.get(
                    "moving_average_window"
                )
            )


            moving_average_text = (
                (
                    f"{format_number(moving_average_window)} "
                    "p\u00e9riodes"
                )

                if (
                    moving_average_window
                    is not None
                )

                else
                None
            )


            candidates = [
                (
                    "P\u00e9riode d'agr\u00e9gation",
                    (
                        time_series_period_label(
                            aggregation_period
                        )

                        if (
                            aggregation_period
                            is not None
                        )

                        else
                        None
                    ),
                ),
                (
                    "Fen\u00eatre moyenne mobile",
                    moving_average_text,
                ),
                (
                    "P\u00e9riodes",
                    metrics.get(
                        "period_count"
                    ),
                ),
                (
                    "Observations",
                    metrics.get(
                        "valid_observations"
                    ),
                ),
                (
                    "Total",
                    metrics.get(
                        "total_revenue"
                    ),
                ),
            ]


            return [
                (
                    label,
                    value,
                )

                for (
                    label,
                    value,
                )
                in candidates

                if (
                    value is not None
                )
            ]


        candidates = [
            (
                "Observations",
                metrics.get(
                    "valid_observations"
                ),
            ),
            (
                "Périodes",
                metrics.get(
                    "period_count"
                ),
            ),
            (
                "Clients distincts",
                metrics.get(
                    "distinct_customers_total"
                ),
            ),
            (
                "Total",
                metrics.get(
                    "total_revenue"
                ),
            ),
        ]


        return [
            (
                label,
                value,
            )

            for (
                label,
                value,
            )
            in candidates

            if (
                value is not None
            )
        ][
            :
            4
        ]


    if (
        family ==
        "ranking"
    ):
        candidates = [
            (
                "Observations",
                metrics.get(
                    "source_observation_count"
                )
                or
                metrics.get(
                    "valid_observations"
                ),
            ),
            (
                "Catégories analysées",
                metrics.get(
                    "available_group_count"
                )
                or
                metrics.get(
                    "group_count"
                ),
            ),
            (
                "Catégories retenues",
                metrics.get(
                    "result_count"
                ),
            ),
            (
                "Agrégation",
                metrics.get(
                    "aggregation_function"
                ),
            ),
        ]


    elif (
        family ==
        "group_comparison"
    ):
        candidates = [
            (
                "Observations",
                metrics.get(
                    "valid_observations"
                )
                or
                metrics.get(
                    "source_observation_count"
                ),
            ),
            (
                "Catégories",
                metrics.get(
                    "group_count"
                )
                or
                metrics.get(
                    "available_group_count"
                ),
            ),
        ]


    else:
        candidates = [
            (
                "Observations",
                metrics.get(
                    "valid_observations"
                )
                or
                metrics.get(
                    "source_observation_count"
                ),
            ),
            (
                "Résultats",
                metrics.get(
                    "result_count"
                ),
            ),
            (
                "Agrégation",
                metrics.get(
                    "aggregation_function"
                ),
            ),
        ]


    return [
        (
            label,
            value,
        )

        for (
            label,
            value,
        )
        in candidates

        if (
            value is not None
        )
    ][
        :
        4
    ]


def ranking_result_table(
    result: dict[
        str,
        Any,
    ],
    styles: dict[
        str,
        ParagraphStyle,
    ],
) -> (
    Table
    | None
):
    rows = (
        ranking_rows(
            result
        )
    )


    if not (
        rows
    ):
        return None


    data: list[
        list[
            Any
        ]
    ] = [
        [
            Paragraph(
                "Rang",
                styles[
                    "label"
                ],
            ),
            Paragraph(
                "Catégorie",
                styles[
                    "label"
                ],
            ),
            Paragraph(
                "Valeur",
                styles[
                    "label"
                ],
            ),
        ]
    ]


    for (
        index,
        (
            label,
            value,
        ),
    ) in enumerate(
        rows,
        start=1,
    ):
        data.append(
            [
                Paragraph(
                    str(
                        index
                    ),
                    styles[
                        "body"
                    ],
                ),
                Paragraph(
                    paragraph_text(
                        label
                    ),
                    styles[
                        "body"
                    ],
                ),
                Paragraph(
                    paragraph_text(
                        format_number(
                            value
                        )
                    ),
                    styles[
                        "body"
                    ],
                ),
            ]
        )


    table = (
        Table(
            data,
            colWidths=[
                18
                *
                mm,
                85
                *
                mm,
                45
                *
                mm,
            ],
            hAlign=
                "LEFT",
        )
    )


    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        0,
                    ),
                    PANEL_BLUE,
                ),
                (
                    "BOX",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    0.5,
                    GRID,
                ),
                (
                    "INNERGRID",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    0.25,
                    GRID,
                ),
                (
                    "VALIGN",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    "MIDDLE",
                ),
                (
                    "LEFTPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    7,
                ),
                (
                    "RIGHTPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    7,
                ),
                (
                    "TOPPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    5,
                ),
            ]
        )
    )


    return table


def analysis_block(
    *,
    detail: dict[
        str,
        Any,
    ],
    index: int,
    styles: dict[
        str,
        ParagraphStyle,
    ],
) -> list[
    Any
]:
    selection = (
        detail.get(
            "selection"
        )
    )


    pipeline_payload = (
        detail.get(
            "pipeline_payload"
        )
    )


    if not isinstance(
        selection,
        dict,
    ) or not isinstance(
        pipeline_payload,
        dict,
    ):
        return []


    (
        result,
        _,
    ) = (
        first_executed_result(
            pipeline_payload
        )
    )


    if (
        result is None
    ):
        return []


    family = (
        analysis_family(
            pipeline_payload,
            result,
        )
    )


    story: list[
        Any
    ] = []


    story.append(
        source_badge(
            str(
                selection.get(
                    "source_type",
                    "",
                )
            ),
            styles,
        )
    )


    story.append(
        Spacer(
            1,
            5,
        )
    )


    objective = clean_text(
        selection.get(
            "objective"
        )
    )


    story.append(
        Paragraph(
            paragraph_text(
                (
                    f"{index}. "
                    +
                    (
                        objective
                        or
                        clean_text(
                            result.get(
                                "title"
                            )
                        )
                    )
                )
            ),
            styles[
                "h2"
            ],
        )
    )


    story.append(
        Paragraph(
            paragraph_text(
                family_label(
                    family
                )
            ),
            styles[
                "small"
            ],
        )
    )


    summary = result.get(
        "summary"
    )


    if isinstance(
        summary,
        list,
    ):
        for line in (
            summary[
                :
                3
            ]
        ):
            story.append(
                Paragraph(
                    paragraph_text(
                        normalize_business_summary(
                            line
                        )
                    ),
                    styles[
                        "body"
                    ],
                )
            )


    metrics = (
        selected_metrics(
            family=
                family,

            result=
                result,
        )
    )


    if (
        metrics
    ):
        story.append(
            metric_table(
                metrics,
                styles,
            )
        )


        story.append(
            Spacer(
                1,
                7,
            )
        )


    if (
        family ==
        "group_comparison"
    ):
        group_summary_table = (
            group_comparison_summary_table(
                result=
                    result,

                styles=
                    styles,
            )
        )


        if (
            group_summary_table
            is not None
        ):
            story.append(
                group_summary_table
            )


            story.append(
                Spacer(
                    1,
                    7,
                )
            )


    if (
        family ==
        "ranking"
    ):
        ranking_table = (
            ranking_result_table(
                result,
                styles,
            )
        )


        if (
            ranking_table is not None
        ):
            story.append(
                Paragraph(
                    "Classement retenu",
                    styles[
                        "label"
                    ],
                )
            )


            story.append(
                ranking_table
            )


            story.append(
                Spacer(
                    1,
                    7,
                )
            )


    chart = (
        result_chart(
            result
        )
    )


    if (
        chart is not None
    ):
        story.append(
            chart
        )


        story.append(
            Spacer(
                1,
                7,
            )
        )


    report_context = (
        pipeline_payload.get(
            "report_context"
        )
    )


    if (
        selection.get(
            "source_type"
        )
        ==
        "document_request"
        and
        isinstance(
            report_context,
            dict,
        )
    ):
        source_filename = (
            display_source_filename(
                report_context.get(
                    "source_filename"
                )
            )
        )

        source_locator = clean_text(
            report_context.get(
                "source_locator"
            )
        )

        page_number = (
            report_context.get(
                "page_number"
            )
        )


        provenance_parts: list[
            str
        ] = []


        if (
            source_filename
        ):
            provenance_parts.append(
                source_filename
            )


        if (
            page_number is not None
        ):
            provenance_parts.append(
                f"page {page_number}"
            )


        elif (
            source_locator
        ):
            provenance_parts.append(
                source_locator
            )


        if (
            provenance_parts
        ):
            story.append(
                Paragraph(
                    paragraph_text(
                        (
                            "Demande issue du document : "
                            +
                            " - ".join(
                                provenance_parts
                            )
                        )
                    ),
                    styles[
                        "small"
                    ],
                )
            )


    warnings = result.get(
        "warnings"
    )


    rendered_warnings = (
        pdf_warning_messages(
            family=
                family,

            warnings=
                warnings,
        )
    )


    if rendered_warnings:
        warning_table = (
            Table(
                [
                    [
                        Paragraph(
                            paragraph_text(
                                (
                                    "Limite de lecture"
                                    if index == 0
                                    else
                                    ""
                                )
                                +
                                (
                                    " - "
                                    if index == 0
                                    else
                                    ""
                                )
                                +
                                warning
                            ),
                            styles[
                                "small"
                            ],
                        )
                    ]

                    for index, warning
                    in enumerate(
                        rendered_warnings
                    )
                ],
                colWidths=[
                    160
                    *
                    mm
                ],
            )
        )


        warning_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        colors.HexColor(
                            "#FFF8EA"
                        ),
                    ),
                    (
                        "BOX",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        0.4,
                        colors.HexColor(
                            "#F0D7A2"
                        ),
                    ),
                    (
                        "LEFTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        7,
                    ),
                    (
                        "RIGHTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        7,
                    ),
                    (
                        "TOPPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        5,
                    ),
                    (
                        "BOTTOMPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        5,
                    ),
                ]
            )
        )


        story.append(
            warning_table
        )


    return story


def simple_table(
    rows: list[
        tuple[
            str,
            object,
        ]
    ],
    styles: dict[
        str,
        ParagraphStyle,
    ],
) -> Table:
    data = [
        [
            Paragraph(
                paragraph_text(
                    label
                ),
                styles[
                    "label"
                ],
            ),
            Paragraph(
                paragraph_text(
                    value
                ),
                styles[
                    "body"
                ],
            ),
        ]

        for (
            label,
            value,
        )
        in rows
    ]


    table = (
        Table(
            data,
            colWidths=[
                48
                *
                mm,
                112
                *
                mm,
            ],
            hAlign=
                "LEFT",
        )
    )


    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    PANEL,
                ),
                (
                    "BOX",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    0.45,
                    GRID,
                ),
                (
                    "INNERGRID",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    0.25,
                    GRID,
                ),
                (
                    "VALIGN",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    7,
                ),
                (
                    "RIGHTPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    7,
                ),
                (
                    "TOPPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    5,
                ),
            ]
        )
    )


    return table


def technical_audit_rows(
    detail: dict[
        str,
        Any,
    ],
) -> list[
    tuple[
        str,
        object,
    ]
]:
    selection = (
        detail.get(
            "selection"
        )
    )

    pipeline_payload = (
        detail.get(
            "pipeline_payload"
        )
    )


    if not isinstance(
        selection,
        dict,
    ) or not isinstance(
        pipeline_payload,
        dict,
    ):
        return []


    (
        result,
        native_tool,
    ) = (
        first_executed_result(
            pipeline_payload
        )
    )


    if (
        result is None
    ):
        return []


    contract = (
        planner_contract_for_result(
            pipeline_payload
        )
    )


    family = (
        analysis_family(
            pipeline_payload,
            result,
        )
    )


    bindings_text = "-"
    aggregation_text = "-"
    ranking_text = "-"


    if isinstance(
        contract,
        dict,
    ):
        bindings = (
            contract.get(
                "bindings",
                [],
            )
        )


        if isinstance(
            bindings,
            list,
        ):
            pairs = [
                (
                    f"{clean_text(binding.get('role'))}="
                    f"{clean_text(binding.get('column'))}"
                )

                for binding
                in bindings

                if isinstance(
                    binding,
                    dict,
                )
            ]


            if (
                pairs
            ):
                bindings_text = ", ".join(
                    pairs
                )


        aggregation = (
            contract.get(
                "aggregation"
            )
        )

        ranking = (
            contract.get(
                "ranking"
            )
        )


        if isinstance(
            aggregation,
            dict,
        ):
            aggregation_text = (
                f"{clean_text(aggregation.get('function'))}; "
                f"groupement={clean_text(aggregation.get('group_by_roles')) or '-'}"
            )


        if isinstance(
            ranking,
            dict,
        ):
            ranking_text = (
                f"{clean_text(ranking.get('order'))}; "
                f"limite={clean_text(ranking.get('limit'))}"
            )


    return [
        (
            "Origine",
            source_type_label(
                clean_text(
                    selection.get(
                        "source_type"
                    )
                )
            ),
        ),
        (
            "Famille",
            family_label(
                family
            ),
        ),
        (
            "Variables",
            bindings_text,
        ),
        (
            "Agrégation",
            aggregation_text,
        ),
        (
            "Classement",
            ranking_text,
        ),
        (
            "Planner",
            pipeline_payload.get(
                "planner_model"
            )
            or
            "-",
        ),
        (
            "Tool model",
            pipeline_payload.get(
                "tool_model"
            )
            or
            "-",
        ),
        (
            "Fonction",
            (
                native_tool.get(
                    "requested_tool"
                )
                if isinstance(
                    native_tool,
                    dict,
                )
                else
                "-"
            )
            or
            "-",
        ),
        (
            "Validation",
            (
                "Validé"
                if (
                    isinstance(
                        native_tool,
                        dict,
                    )
                    and
                    native_tool.get(
                        "validation_status"
                    )
                    ==
                    "validated"
                )
                else
                (
                    native_tool.get(
                        "validation_status"
                    )
                    if isinstance(
                        native_tool,
                        dict,
                    )
                    else
                    "-"
                )
            )
            or
            "-",
        ),
        (
            "analysis_id",
            selection.get(
                "analysis_id"
            )
            or
            "-",
        ),
    ]


# ============================================================
# PUBLIC BUILDER
# ============================================================

def build_selected_report_pdf(
    selection:
        ReportSelectionDetailResponse
        |
        dict[
            str,
            Any,
        ],

    *,
    preparation_context: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None,
) -> bytes:
    if hasattr(
        selection,
        "model_dump",
    ):
        selection_data = (
            selection.model_dump(
                mode="python"
            )
        )


    elif isinstance(
        selection,
        dict,
    ):
        selection_data = (
            selection
        )


    else:
        raise TypeError(
            (
                "selection must be a "
                "ReportSelectionDetailResponse or mapping."
            )
        )


    analyses = (
        selection_data.get(
            "analyses",
            [],
        )
    )


    if not isinstance(
        analyses,
        list,
    ) or not analyses:
        raise ValueError(
            (
                "At least one server-selected analysis "
                "is required to generate the PDF."
            )
        )


    analyses = [
        detail
        for detail
        in analyses
        if isinstance(
            detail,
            dict,
        )
    ]


    workflow_id = clean_text(
        selection_data.get(
            "workflow_id"
        )
    )


    if (
        preparation_context is None
    ):
        preparation_context = (
            build_preparation_context(
                workflow_id
            )
        )


    styles = (
        build_styles()
    )


    buffer = (
        BytesIO()
    )


    document = (
        SimpleDocTemplate(
            buffer,
            pagesize=
                A4,
            rightMargin=
                17
                *
                mm,
            leftMargin=
                17
                *
                mm,
            topMargin=
                18
                *
                mm,
            bottomMargin=
                20
                *
                mm,
            title=
                "DataLens - Rapport d'analyse",
            author=
                "DataLens",
        )
    )


    story: list[
        Any
    ] = []


    # ========================================================
    # 1. COVER + EXECUTIVE SUMMARY
    # ========================================================

    eyebrow_style = ParagraphStyle(
        "SelectedReportCoverEyebrow",
        parent=
            styles[
                "small"
            ],
        fontName=
            "Helvetica-Bold",
        fontSize=
            7.5,
        leading=
            9,
        textColor=
            ACCENT,
    )


    story.append(
        Paragraph(
            "RAPPORT D'ANALYSE",
            eyebrow_style,
        )
    )


    story.append(
        Paragraph(
            "DataLens",
            styles[
                "title"
            ],
        )
    )


    story.append(
        Paragraph(
            (
                "Synthèse des analyses sélectionnées, "
                "validées par Python et exécutées par "
                "le moteur déterministe local."
            ),
            styles[
                "subtitle"
            ],
        )
    )


    prompt_objective = ""


    requested_objectives: list[
        str
    ] = []


    requested_preview_limit = 5


    for detail in (
        analyses
    ):
        selection_item = (
            detail.get(
                "selection"
            )
        )


        if not isinstance(
            selection_item,
            dict,
        ):
            continue


        source_type = clean_text(
            selection_item.get(
                "source_type"
            )
        )


        objective_text = clean_text(
            selection_item.get(
                "objective"
            )
        )


        if not (
            objective_text
        ):
            continue


        if (
            source_type ==
            "initial_request"
            and
            not prompt_objective
        ):
            prompt_objective = (
                objective_text
            )


        if (
            source_type
            in {
                "initial_request",
                "follow_up_prompt",
                "document_request",
            }
            and
            objective_text
            not in requested_objectives
        ):
            requested_objectives.append(
                objective_text
            )


    if (
        prompt_objective
    ):
        story.append(
            Paragraph(
                "OBJECTIF PRINCIPAL",
                eyebrow_style,
            )
        )


        objective_panel = (
            Table(
                [
                    [
                        Paragraph(
                            paragraph_text(
                                prompt_objective
                            ),
                            styles[
                                "callout"
                            ],
                        )
                    ]
                ],
                colWidths=[
                    160
                    *
                    mm
                ],
            )
        )


        objective_panel.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        PANEL_BLUE,
                    ),
                    (
                        "BOX",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        0.5,
                        colors.HexColor(
                            "#CDDCF6"
                        ),
                    ),
                    (
                        "LEFTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        9,
                    ),
                    (
                        "RIGHTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        9,
                    ),
                    (
                        "TOPPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        7,
                    ),
                    (
                        "BOTTOMPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        7,
                    ),
                ]
            )
        )


        story.append(
            objective_panel
        )


        story.append(
            Spacer(
                1,
                10,
            )
        )


    elif (
        requested_objectives
    ):
        story.append(
            Paragraph(
                (
                    "ANALYSES DEMAND\u00c9ES"
                    f" \u00b7 {len(requested_objectives)} RETENUES"
                ),
                eyebrow_style,
            )
        )


        requested_table = (
            Table(
                [
                    [
                        Paragraph(
                            paragraph_text(
                                (
                                    f"{index}. "
                                    f"{objective_text}"
                                )
                            ),
                            styles[
                                "body"
                            ],
                        )
                    ]

                    for (
                        index,
                        objective_text,
                    )
                    in enumerate(
                        requested_objectives[
                            :
                            requested_preview_limit
                        ],
                        start=1,
                    )
                ],
                colWidths=[
                    160
                    *
                    mm
                ],
            )
        )


        requested_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        PANEL_BLUE,
                    ),
                    (
                        "BOX",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        0.45,
                        colors.HexColor(
                            "#CDDCF6"
                        ),
                    ),
                    (
                        "INNERGRID",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        0.25,
                        colors.HexColor(
                            "#DCE7FA"
                        ),
                    ),
                    (
                        "LEFTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        8,
                    ),
                    (
                        "RIGHTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        8,
                    ),
                    (
                        "TOPPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        6,
                    ),
                    (
                        "BOTTOMPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        6,
                    ),
                ]
            )
        )


        story.append(
            requested_table
        )


        hidden_requested_count = max(
            len(
                requested_objectives
            )
            -
            requested_preview_limit,
            0,
        )


        if (
            hidden_requested_count
            >
            0
        ):
            story.append(
                Spacer(
                    1,
                    4,
                )
            )


            story.append(
                Paragraph(
                    paragraph_text(
                        (
                            f"Aper\u00e7u de "
                            f"{min(requested_preview_limit, len(requested_objectives))} "
                            f"analyse(s) sur "
                            f"{len(requested_objectives)}. "
                            f"Les {hidden_requested_count} autre(s) "
                            f"sont d\u00e9taill\u00e9es dans le rapport."
                        )
                    ),
                    styles[
                        "body"
                    ],
                )
            )


        story.append(
            Spacer(
                1,
                10,
            )
        )


    all_insights = [
        deterministic_insight(
            detail
        )
        for detail
        in analyses
    ]


    available_insights = [
        insight
        for insight
        in all_insights
        if insight
    ]


    insight_preview_limit = 4


    insights = (
        available_insights[
            :
            insight_preview_limit
        ]
    )


    if (
        insights
    ):
        story.append(
            Paragraph(
                (
                    "PRINCIPAUX R\u00c9SULTATS"
                    f" \u00b7 {len(insights)} SUR "
                    f"{len(available_insights)}"
                ),
                eyebrow_style,
            )
        )


        if (
            len(
                available_insights
            )
            >
            len(
                insights
            )
        ):
            story.append(
                Paragraph(
                    paragraph_text(
                        (
                            f"{len(insights)} r\u00e9sultat(s) cl\u00e9(s) "
                            f"sont mis en avant sur cette page. "
                            f"Les r\u00e9sultats complets des "
                            f"{len(available_insights)} analyse(s) "
                            f"sont d\u00e9taill\u00e9s dans le rapport."
                        )
                    ),
                    styles[
                        "body"
                    ],
                )
            )


            story.append(
                Spacer(
                    1,
                    4,
                )
            )


        insight_table = (
            Table(
                [
                    [
                        Paragraph(
                            paragraph_text(
                                f"{index}. {insight}"
                            ),
                            styles[
                                "body"
                            ],
                        )
                    ]
                    for (
                        index,
                        insight,
                    )
                    in enumerate(
                        insights,
                        start=1,
                    )
                ],
                colWidths=[
                    160
                    *
                    mm
                ],
            )
        )


        insight_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        PANEL,
                    ),
                    (
                        "BOX",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        0.45,
                        GRID,
                    ),
                    (
                        "INNERGRID",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        0.25,
                        GRID,
                    ),
                    (
                        "LEFTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        8,
                    ),
                    (
                        "RIGHTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        8,
                    ),
                    (
                        "TOPPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        6,
                    ),
                    (
                        "BOTTOMPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        6,
                    ),
                ]
            )
        )


        story.append(
            insight_table
        )


    story.append(
        Spacer(
            1,
            9,
        )
    )


    story.append(
        Paragraph(
            (
                "Rapport généré localement le "
                +
                datetime.now()
                .strftime(
                    "%d/%m/%Y à %H:%M"
                )
                +
                "."
            ),
            styles[
                "small"
            ],
        )
    )


    story.append(
        PageBreak()
    )


    # ========================================================
    # 2. BUSINESS ANALYSES
    # ========================================================

    story.append(
        Paragraph(
            "Analyses retenues",
            styles[
                "h1"
            ],
        )
    )


    story.append(
        Paragraph(
            (
                "Les résultats ci-dessous correspondent "
                "exactement à la sélection du rapport. "
                "Les analyses exploratoires non retenues "
                "ne sont pas exportées."
            ),
            styles[
                "subtitle"
            ],
        )
    )


    previous_compact = False


    for (
        index,
        detail,
    ) in enumerate(
        analyses,
        start=1,
    ):
        block = (
            analysis_block(
                detail=
                    detail,
                index=
                    index,
                styles=
                    styles,
            )
        )


        if not (
            block
        ):
            continue


        pipeline_payload = (
            detail.get(
                "pipeline_payload"
            )
        )


        current_compact = False


        if isinstance(
            pipeline_payload,
            dict,
        ):
            (
                current_result,
                _,
            ) = (
                first_executed_result(
                    pipeline_payload
                )
            )


            if isinstance(
                current_result,
                dict,
            ):
                current_family = (
                    analysis_family(
                        pipeline_payload,
                        current_result,
                    )
                )


                current_chart_type = (
                    clean_text(
                        current_result.get(
                            "chart_type"
                        )
                    )
                    .lower()
                )


                current_compact = (
                    current_family ==
                    "descriptive_metric"
                    or
                    current_chart_type
                    in {
                        "",
                        "none",
                        "metric",
                    }
                )


        if (
            index >
            1
        ):
            if (
                current_compact
                and
                previous_compact
            ):
                story.append(
                    Spacer(
                        1,
                        16,
                    )
                )


            else:
                story.append(
                    PageBreak()
                )


        story.append(
            KeepTogether(
                block
            )
        )


        previous_compact = (
            current_compact
        )


    story.append(
        PageBreak()
    )


    # ========================================================
    # 3. UNRESOLVED DOCUMENT REQUESTS
    # ========================================================

    unresolved_section = (
        build_unresolved_requested_section(
            workflow_id=
                workflow_id,

            styles=
                styles,
        )
    )


    if (
        unresolved_section
    ):
        story.extend(
            unresolved_section
        )


        story.append(
            PageBreak()
        )


    # ========================================================
    # 4. METHODOLOGY + PREPARATION
    # ========================================================

    story.append(
        Paragraph(
            "Méthodologie et préparation",
            styles[
                "h1"
            ],
        )
    )


    story.append(
        Paragraph(
            (
                "DataLens sépare la compréhension de la demande, "
                "la validation du contrat analytique et le calcul "
                "statistique. Les modèles locaux ne disposent pas "
                "d'un accès libre à Python."
            ),
            styles[
                "body"
            ],
        )
    )


    story.append(
        Spacer(
            1,
            5,
        )
    )


    story.append(
        Paragraph(
            "Préparation certifiée",
            styles[
                "h2"
            ],
        )
    )


    if (
        isinstance(
            preparation_context,
            dict,
        )
        and
        preparation_context.get(
            "available"
        )
    ):
        story.append(
            simple_table(
                [
                    (
                        "État final",
                        (
                            "Prêt pour l'analyse"
                            if preparation_context.get(
                                "ready_for_analysis"
                            )
                            else
                            "Non prêt"
                        ),
                    ),
                    (
                        "Jeux de données finaux",
                        preparation_context.get(
                            "dataset_count"
                        )
                        or
                        "-",
                    ),
                    (
                        "Lignes analysées",
                        format_number(
                            preparation_context.get(
                                "total_rows"
                            )
                        )
                        or
                        "-",
                    ),
                    (
                        "Révision préparation",
                        preparation_context.get(
                            "session_revision"
                        )
                        or
                        "-",
                    ),
                ],
                styles,
            )
        )


        stages = (
            preparation_context.get(
                "stages",
                [],
            )
        )


        if isinstance(
            stages,
            list,
        ) and stages:
            story.append(
                Spacer(
                    1,
                    7,
                )
            )


            stage_data = [
                [
                    Paragraph(
                        "Étape",
                        styles[
                            "label"
                        ],
                    ),
                    Paragraph(
                        "Statut",
                        styles[
                            "label"
                        ],
                    ),
                ]
            ]


            for stage in (
                stages
            ):
                if not isinstance(
                    stage,
                    dict,
                ):
                    continue


                stage_data.append(
                    [
                        Paragraph(
                            paragraph_text(
                                stage_label(
                                    clean_text(
                                        stage.get(
                                            "stage"
                                        )
                                    )
                                )
                            ),
                            styles[
                                "body"
                            ],
                        ),
                        Paragraph(
                            paragraph_text(
                                status_label(
                                    clean_text(
                                        stage.get(
                                            "status"
                                        )
                                    ),

                                    stage=
                                        clean_text(
                                            stage.get(
                                                "stage"
                                            )
                                        ),

                                    required=
                                        bool(
                                            stage.get(
                                                "required",
                                                False,
                                            )
                                        ),

                                    materialized=
                                        bool(
                                            stage.get(
                                                "materialized",
                                                False,
                                            )
                                        ),
                                )
                            ),
                            styles[
                                "body"
                            ],
                        ),
                    ]
                )


            stage_table = (
                Table(
                    stage_data,
                    colWidths=[
                        80
                        *
                        mm,
                        80
                        *
                        mm,
                    ],
                    hAlign=
                        "LEFT",
                )
            )


            stage_table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (
                                0,
                                0,
                            ),
                            (
                                -1,
                                0,
                            ),
                            PANEL_BLUE,
                        ),
                        (
                            "BOX",
                            (
                                0,
                                0,
                            ),
                            (
                                -1,
                                -1,
                            ),
                            0.45,
                            GRID,
                        ),
                        (
                            "INNERGRID",
                            (
                                0,
                                0,
                            ),
                            (
                                -1,
                                -1,
                            ),
                            0.25,
                            GRID,
                        ),
                        (
                            "LEFTPADDING",
                            (
                                0,
                                0,
                            ),
                            (
                                -1,
                                -1,
                            ),
                            7,
                        ),
                        (
                            "RIGHTPADDING",
                            (
                                0,
                                0,
                            ),
                            (
                                -1,
                                -1,
                            ),
                            7,
                        ),
                        (
                            "TOPPADDING",
                            (
                                0,
                                0,
                            ),
                            (
                                -1,
                                -1,
                            ),
                            5,
                        ),
                        (
                            "BOTTOMPADDING",
                            (
                                0,
                                0,
                            ),
                            (
                                -1,
                                -1,
                            ),
                            5,
                        ),
                    ]
                )
            )


            story.append(
                stage_table
            )


        datasets = (
            preparation_context.get(
                "datasets",
                [],
            )
        )


        if isinstance(
            datasets,
            list,
        ) and datasets:
            story.append(
                Spacer(
                    1,
                    7,
                )
            )


            story.append(
                Paragraph(
                    "Sortie(s) analysée(s)",
                    styles[
                        "h2"
                    ],
                )
            )


            for dataset in (
                datasets
            ):
                if not isinstance(
                    dataset,
                    dict,
                ):
                    continue


                line = (
                    f"{dataset.get('filename') or dataset.get('dataset_id')} "
                    f"- étape {stage_label(clean_text(dataset.get('stage')))}"
                )


                if (
                    dataset.get(
                        "column_count"
                    )
                    is not None
                ):
                    line += (
                        " - "
                        f"{format_number(dataset.get('column_count'))} colonnes"
                    )


                story.append(
                    Paragraph(
                        paragraph_text(
                            line
                        ),
                        styles[
                            "small"
                        ],
                    )
                )


    else:
        story.append(
            Paragraph(
                (
                    "Les métadonnées détaillées de préparation "
                    "n'étaient pas disponibles lors de cet export. "
                    "La sélection analytique server-owned reste "
                    "la source du document."
                ),
                styles[
                    "small"
                ],
            )
        )


    story.append(
        Spacer(
            1,
            10,
        )
    )


    story.append(
        Paragraph(
            "Chaîne analytique",
            styles[
                "h2"
            ],
        )
    )


    story.append(
        simple_table(
            [
                (
                    "Planification",
                    (
                        "Le modèle local propose un contrat "
                        "analytique structuré."
                    ),
                ),
                (
                    "Validation",
                    (
                        "Python vérifie les datasets, colonnes, "
                        "types et invariants du contrat."
                    ),
                ),
                (
                    "Routage d'outil",
                    (
                        "Le modèle de routage choisit uniquement "
                        "une fonction autorisée."
                    ),
                ),
                (
                    "Calcul",
                    (
                        "Python exécute le calcul statistique "
                        "déterministe."
                    ),
                ),
                (
                    "Export",
                    (
                        "Le PDF relit la sélection et les résultats "
                        "persistés côté serveur."
                    ),
                ),
            ],
            styles,
        )
    )


    story.append(
        PageBreak()
    )


    # ========================================================
    # 5. TECHNICAL APPENDIX
    # ========================================================

    story.append(
        Paragraph(
            "Annexe technique",
            styles[
                "h1"
            ],
        )
    )


    story.append(
        Paragraph(
            (
                "Les informations d'audit sont regroupées ici "
                "afin de préserver une lecture métier dans le "
                "corps du rapport."
            ),
            styles[
                "subtitle"
            ],
        )
    )


    story.append(
        simple_table(
            [
                (
                    "Workflow",
                    workflow_id
                    or
                    "-",
                ),
                (
                    "Révision sélection",
                    selection_data.get(
                        "revision"
                    )
                    or
                    "-",
                ),
                (
                    "Analyses retenues",
                    len(
                        analyses
                    ),
                ),
                (
                    "Version export",
                    SELECTED_REPORT_PDF_RULE_VERSION,
                ),
                (
                    "Données brutes",
                    "Non incluses dans le PDF",
                ),
            ],
            styles,
        )
    )


    for (
        index,
        detail,
    ) in enumerate(
        analyses,
        start=1,
    ):
        selection_item = (
            detail.get(
                "selection"
            )
        )


        objective = ""


        if isinstance(
            selection_item,
            dict,
        ):
            objective = clean_text(
                selection_item.get(
                    "objective"
                )
            )


        rows = (
            technical_audit_rows(
                detail
            )
        )


        audit_block: list[
            Any
        ] = [
            Spacer(
                1,
                10,
            ),
            Paragraph(
                paragraph_text(
                    f"{index}. {objective or 'Analyse'}"
                ),
                styles[
                    "h2"
                ],
            ),
        ]


        if (
            rows
        ):
            audit_block.append(
                simple_table(
                    rows,
                    styles,
                )
            )


        story.append(
            KeepTogether(
                audit_block
            )
        )


    document.build(
        story,
        onFirstPage=
            page_decorator,
        onLaterPages=
            page_decorator,
    )


    return (
        buffer.getvalue()
    )


def build_selected_report_filename() -> str:
    date = (
        datetime.now()
        .strftime(
            "%Y-%m-%d"
        )
    )


    return (
        f"datalens-rapport-{date}.pdf"
    )
