from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

from reportlab.graphics.shapes import (
    Circle,
    Drawing,
    Line,
    PolyLine,
    Rect,
    String,
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PDF_EXPORT_RULE_VERSION = "pdf_export_v0.2"


# ============================================================
# COLORS
# ============================================================

INK = colors.HexColor("#162033")
MUTED = colors.HexColor("#5F6B7A")
ACCENT = colors.HexColor("#356FD6")
ACCENT_LIGHT = colors.HexColor("#EAF1FF")
GRID = colors.HexColor("#D9E1EC")
PANEL = colors.HexColor("#F6F8FB")
SUCCESS = colors.HexColor("#147D64")
WARNING = colors.HexColor("#9A6412")
DANGER = colors.HexColor("#A23A45")
WHITE = colors.white


# ============================================================
# TEXT / VALUE HELPERS
# ============================================================

def clean_text(value: object) -> str:
    text = str(value or "")

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u202f": " ",
        "\u00a0": " ",
        "\u2192": "->",
        "\u2197": "->",
        "\u0153": "oe",
        "\u0152": "OE",
    }

    for source, target in replacements.items():
        text = text.replace(source, target)

    return " ".join(text.split())


def to_mapping(value: object | None) -> dict[str, Any] | None:
    if value is None:
        return None

    if isinstance(value, dict):
        return value

    model_dump = getattr(value, "model_dump", None)

    if callable(model_dump):
        dumped = model_dump(mode="python")

        if isinstance(dumped, dict):
            return dumped

    return None


def list_of_mappings(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    result: list[dict[str, Any]] = []

    for item in value:
        mapping = to_mapping(item)

        if mapping is not None:
            result.append(mapping)

    return result


def safe_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        number = float(value)

        if number == number and number not in {float("inf"), float("-inf")}:
            return number

    return None


def format_number(value: object) -> str:
    number = safe_number(value)

    if number is None:
        return clean_text(value)

    absolute = abs(number)

    if absolute >= 1_000_000:
        return f"{number / 1_000_000:.2f} M".replace(".00", "")

    if absolute >= 1_000:
        return f"{number / 1_000:.1f} k".replace(".0 k", " k")

    if float(number).is_integer():
        return f"{int(number):,}".replace(",", " ")

    return f"{number:.3f}".rstrip("0").rstrip(".")


def format_percent(value: object) -> str:
    number = safe_number(value)

    if number is None:
        return clean_text(value)

    if abs(number) <= 1.0:
        number *= 100.0

    return f"{number:.1f}%"


def first_non_empty(*values: object) -> str:
    for value in values:
        text = clean_text(value)

        if text:
            return text

    return "-"


def truncate(value: object, max_length: int = 42) -> str:
    text = clean_text(value)

    if len(text) <= max_length:
        return text

    return text[: max_length - 1].rstrip() + "..."


# ============================================================
# STYLES
# ============================================================

def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "DataLensTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=29,
            textColor=INK,
            spaceAfter=8,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "DataLensSubtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=MUTED,
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "DataLensH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=INK,
            spaceBefore=8,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "DataLensH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=16,
            textColor=INK,
            spaceBefore=5,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "DataLensBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=INK,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "DataLensSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10.5,
            textColor=MUTED,
            spaceAfter=3,
        ),
        "label": ParagraphStyle(
            "DataLensLabel",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9.5,
            textColor=MUTED,
            uppercase=True,
            spaceAfter=2,
        ),
        "metric": ParagraphStyle(
            "DataLensMetric",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            textColor=INK,
        ),
        "quote": ParagraphStyle(
            "DataLensQuote",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.8,
            leading=12.5,
            textColor=MUTED,
            leftIndent=7,
            borderColor=GRID,
            borderWidth=0,
            borderPadding=0,
            spaceBefore=2,
            spaceAfter=4,
        ),
    }


# ============================================================
# PAGE CHROME
# ============================================================

def page_decorator(canvas, document) -> None:
    canvas.saveState()

    width, height = A4

    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.5)
    canvas.line(
        document.leftMargin,
        13 * mm,
        width - document.rightMargin,
        13 * mm,
    )

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        document.leftMargin,
        8.8 * mm,
        "DataLens - rapport analytique local",
    )

    canvas.drawRightString(
        width - document.rightMargin,
        8.8 * mm,
        f"Page {document.page}",
    )

    canvas.restoreState()


# ============================================================
# TABLE / CARD HELPERS
# ============================================================

def metric_table(
    metrics: list[tuple[str, object]],
    styles: dict[str, ParagraphStyle],
) -> Table:
    cells = []

    for label, value in metrics:
        cells.append(
            [
                Paragraph(clean_text(label), styles["label"]),
                Paragraph(format_number(value), styles["metric"]),
            ]
        )

    table = Table(
        cells,
        colWidths=[38 * mm, 32 * mm],
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PANEL),
                ("BOX", (0, 0), (-1, -1), 0.5, GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    return table


def status_badge(
    status: str,
    styles: dict[str, ParagraphStyle],
) -> Table:
    normalized = clean_text(status).lower()

    if normalized in {"complete", "ready", "descriptive_only"}:
        background = colors.HexColor("#E8F5F1")
        text_color = SUCCESS
    elif normalized in {"blocked", "failed", "not_executed"}:
        background = colors.HexColor("#FBEAEC")
        text_color = DANGER
    else:
        background = colors.HexColor("#FFF4DF")
        text_color = WARNING

    style = ParagraphStyle(
        "StatusBadge",
        parent=styles["small"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9,
        textColor=text_color,
    )

    table = Table(
        [[Paragraph(clean_text(status).upper(), style)]],
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.4, background),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )

    return table


# ============================================================
# CHART HELPERS
# ============================================================

def chart_value(datum: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = safe_number(datum.get(key))

        if value is not None:
            return value

    return None


def chart_label(datum: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = datum.get(key)

        if value is None:
            continue

        text = clean_text(value)

        if text:
            return text

    return None


def empty_chart(message: str) -> Drawing:
    drawing = Drawing(500, 70)
    drawing.add(Rect(0, 0, 500, 70, fillColor=PANEL, strokeColor=GRID))
    drawing.add(
        String(
            12,
            30,
            clean_text(message),
            fontName="Helvetica",
            fontSize=8,
            fillColor=MUTED,
        )
    )
    return drawing


def line_chart(data: list[dict[str, Any]]) -> Drawing:
    points = []

    for index, datum in enumerate(data):
        value = chart_value(datum, "value", "median")

        if value is not None:
            points.append((index, value, datum))

    if len(points) < 2:
        return empty_chart("Pas assez de points pour la courbe.")

    drawing = Drawing(500, 205)
    left, bottom, width, height = 42, 28, 440, 150

    values = [point[1] for point in points]
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum or 1.0

    drawing.add(Line(left, bottom, left, bottom + height, strokeColor=GRID))
    drawing.add(Line(left, bottom, left + width, bottom, strokeColor=GRID))

    for ratio in (0, 0.5, 1):
        y = bottom + ratio * height
        drawing.add(Line(left, y, left + width, y, strokeColor=GRID, strokeWidth=0.35))
        value = minimum + ratio * span
        drawing.add(
            String(
                2,
                y - 2,
                format_number(value),
                fontName="Helvetica",
                fontSize=6.5,
                fillColor=MUTED,
            )
        )

    projected = []

    for offset, (_, value, _) in enumerate(points):
        x = left + (offset / max(len(points) - 1, 1)) * width
        y = bottom + ((value - minimum) / span) * height
        projected.extend([x, y])

    drawing.add(
        PolyLine(
            projected,
            strokeColor=ACCENT,
            strokeWidth=2.0,
            fillColor=None,
        )
    )

    moving = []

    for offset, (_, _, datum) in enumerate(points):
        moving_value = chart_value(datum, "moving_average")

        if moving_value is None:
            continue

        x = left + (offset / max(len(points) - 1, 1)) * width
        y = bottom + ((moving_value - minimum) / span) * height
        moving.extend([x, y])

    if len(moving) >= 4:
        drawing.add(
            PolyLine(
                moving,
                strokeColor=MUTED,
                strokeWidth=1.5,
                fillColor=None,
                strokeDashArray=[5, 4],
            )
        )

    start_label = chart_label(points[0][2], "period") or "Début"
    end_label = chart_label(points[-1][2], "period") or "Fin"

    drawing.add(String(left, 10, truncate(start_label, 18), fontSize=6.5, fillColor=MUTED))
    drawing.add(
        String(
            left + width - 72,
            10,
            truncate(end_label, 18),
            fontSize=6.5,
            fillColor=MUTED,
        )
    )

    return drawing


def bar_chart(data: list[dict[str, Any]]) -> Drawing:
    rows = []

    for index, datum in enumerate(data[:12]):
        label = chart_label(datum, "category", "group", "label", "x") or str(index + 1)
        value = chart_value(datum, "value", "count", "median", "mean")

        if value is not None:
            rows.append((label, value))

    if not rows:
        return empty_chart("Aucune valeur exploitable pour le graphique.")

    drawing = Drawing(500, max(120, 28 + len(rows) * 19))
    left, width = 135, 315
    max_value = max(abs(value) for _, value in rows) or 1.0
    row_height = 18
    top = drawing.height - 22

    for index, (label, value) in enumerate(rows):
        y = top - index * row_height
        drawing.add(
            String(
                4,
                y + 2,
                truncate(label, 30),
                fontName="Helvetica",
                fontSize=7,
                fillColor=INK,
            )
        )
        bar_width = max(1.5, abs(value) / max_value * width)
        drawing.add(
            Rect(
                left,
                y,
                bar_width,
                8,
                fillColor=ACCENT,
                strokeColor=None,
            )
        )
        drawing.add(
            String(
                min(left + bar_width + 4, 466),
                y + 1,
                format_number(value),
                fontName="Helvetica",
                fontSize=6.5,
                fillColor=MUTED,
            )
        )

    return drawing


def scatter_chart(data: list[dict[str, Any]]) -> Drawing:
    points = []

    for datum in data[:700]:
        x = chart_value(datum, "x")
        y = chart_value(datum, "y")

        if x is not None and y is not None:
            points.append((x, y))

    if len(points) < 2:
        return empty_chart("Pas assez de points pour le nuage.")

    drawing = Drawing(500, 210)
    left, bottom, width, height = 46, 28, 430, 158

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0

    drawing.add(Line(left, bottom, left, bottom + height, strokeColor=GRID))
    drawing.add(Line(left, bottom, left + width, bottom, strokeColor=GRID))

    for x_value, y_value in points:
        x = left + ((x_value - x_min) / x_span) * width
        y = bottom + ((y_value - y_min) / y_span) * height
        drawing.add(Circle(x, y, 1.25, fillColor=ACCENT, strokeColor=None, fillOpacity=0.45))

    drawing.add(String(left, 8, format_number(x_min), fontSize=6, fillColor=MUTED))
    drawing.add(String(left + width - 40, 8, format_number(x_max), fontSize=6, fillColor=MUTED))
    drawing.add(String(2, bottom, format_number(y_min), fontSize=6, fillColor=MUTED))
    drawing.add(String(2, bottom + height - 2, format_number(y_max), fontSize=6, fillColor=MUTED))

    return drawing


def heatmap_chart(data: list[dict[str, Any]]) -> Drawing:
    cells = []

    for datum in data:
        x = chart_label(datum, "x")
        y = chart_label(datum, "y")
        count = chart_value(datum, "count")

        if x and y and count is not None:
            cells.append((x, y, count))

    if not cells:
        return empty_chart("Aucune cellule de contingence exploitable.")

    x_values = list(dict.fromkeys(cell[0] for cell in cells))
    y_values = list(dict.fromkeys(cell[1] for cell in cells))
    maximum = max(cell[2] for cell in cells) or 1.0

    drawing = Drawing(500, max(150, 70 + len(y_values) * 42))
    left, bottom = 92, 38
    plot_width = 390
    plot_height = drawing.height - 72
    cell_width = plot_width / max(len(x_values), 1)
    cell_height = plot_height / max(len(y_values), 1)
    lookup = {(x, y): count for x, y, count in cells}

    for x_index, x_value in enumerate(x_values):
        drawing.add(
            String(
                left + (x_index + 0.5) * cell_width - 10,
                drawing.height - 22,
                truncate(x_value, 12),
                fontSize=6.5,
                fillColor=MUTED,
            )
        )

    for y_index, y_value in enumerate(y_values):
        y = bottom + (len(y_values) - y_index - 1) * cell_height
        drawing.add(String(2, y + cell_height / 2 - 2, truncate(y_value, 18), fontSize=6.5, fillColor=MUTED))

        for x_index, x_value in enumerate(x_values):
            count = lookup.get((x_value, y_value), 0.0)
            intensity = max(0.10, min(0.95, count / maximum))
            fill = colors.Color(
                ACCENT.red,
                ACCENT.green,
                ACCENT.blue,
                alpha=intensity,
            )
            x = left + x_index * cell_width
            drawing.add(
                Rect(
                    x + 1.5,
                    y + 1.5,
                    max(cell_width - 3, 1),
                    max(cell_height - 3, 1),
                    fillColor=fill,
                    strokeColor=WHITE,
                    strokeWidth=0.5,
                )
            )
            drawing.add(
                String(
                    x + cell_width / 2 - 8,
                    y + cell_height / 2 - 2,
                    format_number(count),
                    fontName="Helvetica-Bold",
                    fontSize=6.5,
                    fillColor=INK,
                )
            )

    return drawing


def boxplot_chart(data: list[dict[str, Any]]) -> Drawing:
    groups = []

    for datum in data:
        label = chart_label(datum, "group")
        minimum = chart_value(datum, "min")
        q1 = chart_value(datum, "q1")
        median = chart_value(datum, "median")
        q3 = chart_value(datum, "q3")
        maximum = chart_value(datum, "max")

        if label and None not in {minimum, q1, median, q3, maximum}:
            groups.append((label, minimum, q1, median, q3, maximum))

    if not groups:
        return empty_chart("Aucun résumé de distribution exploitable.")

    all_values = [number for group in groups for number in group[1:]]
    minimum = min(all_values)
    maximum = max(all_values)
    span = maximum - minimum or 1.0

    drawing = Drawing(500, max(135, 48 + len(groups) * 36))
    left, width = 102, 370
    top = drawing.height - 28

    def project(value: float) -> float:
        return left + ((value - minimum) / span) * width

    drawing.add(Line(left, 20, left + width, 20, strokeColor=GRID))

    for index, (label, low, q1, median, q3, high) in enumerate(groups):
        y = top - index * 35
        drawing.add(String(2, y - 2, truncate(label, 20), fontSize=7, fillColor=INK))
        drawing.add(Line(project(low), y, project(high), y, strokeColor=MUTED, strokeWidth=1.2))
        drawing.add(Rect(project(q1), y - 7, max(project(q3) - project(q1), 1), 14, fillColor=ACCENT_LIGHT, strokeColor=ACCENT))
        drawing.add(Line(project(median), y - 9, project(median), y + 9, strokeColor=ACCENT, strokeWidth=2))

    drawing.add(String(left, 6, format_number(minimum), fontSize=6.5, fillColor=MUTED))
    drawing.add(String(left + width - 35, 6, format_number(maximum), fontSize=6.5, fillColor=MUTED))

    return drawing


def lorenz_chart(data: list[dict[str, Any]]) -> Drawing:
    points = []

    for datum in data:
        population = chart_value(datum, "population_share")
        revenue = chart_value(datum, "revenue_share")

        if population is not None and revenue is not None:
            points.append((population, revenue))

    if len(points) < 2:
        return empty_chart("Pas assez de points pour la courbe de Lorenz.")

    drawing = Drawing(500, 230)
    left, bottom, size = 60, 35, 165

    for ratio in (0, 0.25, 0.5, 0.75, 1):
        x = left + ratio * size
        y = bottom + ratio * size
        drawing.add(Line(left, y, left + size, y, strokeColor=GRID, strokeWidth=0.35))
        drawing.add(Line(x, bottom, x, bottom + size, strokeColor=GRID, strokeWidth=0.35))
        drawing.add(String(x - 7, 16, format_percent(ratio), fontSize=6, fillColor=MUTED))

    drawing.add(
        Line(
            left,
            bottom,
            left + size,
            bottom + size,
            strokeColor=MUTED,
            strokeWidth=1,
            strokeDashArray=[5, 4],
        )
    )

    projected = []

    for population, revenue in points:
        projected.extend(
            [
                left + max(0, min(1, population)) * size,
                bottom + max(0, min(1, revenue)) * size,
            ]
        )

    drawing.add(
        PolyLine(
            projected,
            strokeColor=ACCENT,
            strokeWidth=2.2,
            fillColor=None,
        )
    )

    drawing.add(String(260, 170, "Courbe observée", fontName="Helvetica-Bold", fontSize=8, fillColor=INK))
    drawing.add(Line(260, 160, 300, 160, strokeColor=ACCENT, strokeWidth=2.2))
    drawing.add(String(260, 135, "Égalité parfaite", fontSize=8, fillColor=MUTED))
    drawing.add(Line(260, 125, 300, 125, strokeColor=MUTED, strokeDashArray=[5, 4]))
    drawing.add(String(260, 82, "X : part cumulée des clients", fontSize=7.5, fillColor=MUTED))
    drawing.add(String(260, 66, "Y : part cumulée du CA", fontSize=7.5, fillColor=MUTED))

    return drawing


def finding_chart(finding: dict[str, Any]) -> Drawing | None:
    chart_type = clean_text(finding.get("chart_type")).lower()
    data = list_of_mappings(finding.get("chart_data"))

    if not data:
        return None

    if chart_type in {"line", "line_band"}:
        return line_chart(data)

    if chart_type in {"bar", "grouped_summary"}:
        return bar_chart(data)

    if chart_type == "scatter":
        return scatter_chart(data)

    if chart_type == "heatmap":
        return heatmap_chart(data)

    if chart_type == "boxplot":
        return boxplot_chart(data)

    if chart_type == "lorenz":
        return lorenz_chart(data)

    return None


# ============================================================
# REPORT SECTIONS
# ============================================================

def requested_finding_block(
    finding: dict[str, Any],
    index: int,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    story: list[Any] = []

    title = first_non_empty(
        finding.get("title"),
        finding.get("request_text"),
        f"Analyse demandée {index + 1}",
    )

    status = first_non_empty(
        finding.get("execution_status"),
        "résultat",
    )

    story.append(Paragraph(f"{index + 1}. {clean_text(title)}", styles["h2"]))
    story.append(status_badge(status, styles))
    story.append(Spacer(1, 4))

    summary = finding.get("summary")

    if isinstance(summary, list):
        for item in summary[:4]:
            story.append(Paragraph(clean_text(item), styles["body"]))

    metrics = to_mapping(finding.get("metrics")) or {}
    descriptive = to_mapping(finding.get("descriptive_statistics")) or {}

    selected_metrics: list[tuple[str, object]] = []

    metric_candidates = [
        ("Observations", metrics.get("valid_observations") or finding.get("sample_size")),
        ("Spearman rho", descriptive.get("spearman_rho")),
        ("Pearson r", descriptive.get("pearson_r")),
        ("Gini", metrics.get("gini_coefficient")),
        ("Transactions", metrics.get("transaction_count")),
        ("Produits vendus", metrics.get("products_sold_count")),
        ("Références distinctes", metrics.get("distinct_products_sold") or metrics.get("reference_count")),
        ("Clients distincts", metrics.get("distinct_customers_total") or metrics.get("customer_count")),
        ("Chiffre d'affaires", metrics.get("total_revenue")),
        ("Catégories", metrics.get("category_count")),
    ]

    for label, value in metric_candidates:
        if value is not None:
            selected_metrics.append((label, value))

        if len(selected_metrics) >= 4:
            break

    if selected_metrics:
        story.append(metric_table(selected_metrics, styles))
        story.append(Spacer(1, 6))

    chart = finding_chart(finding)

    if chart is not None:
        story.append(chart)
        story.append(Spacer(1, 6))

    source_filename = clean_text(finding.get("source_filename"))
    source_locator = clean_text(finding.get("source_locator"))
    evidence = clean_text(finding.get("evidence_quote"))

    if source_filename or source_locator:
        story.append(
            Paragraph(
                f"Source documentaire : {source_filename or '-'}"
                + (f" - {source_locator}" if source_locator else ""),
                styles["small"],
            )
        )

    if evidence:
        story.append(Paragraph(f"Preuve : {evidence}", styles["quote"]))

    caveats = finding.get("caveats") or finding.get("limitations")

    if isinstance(caveats, list) and caveats:
        story.append(
            Paragraph(
                "Limite principale : " + clean_text(caveats[0]),
                styles["small"],
            )
        )

    story.append(Spacer(1, 8))

    return story


def exploratory_finding_block(
    finding: dict[str, Any],
    index: int,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    story: list[Any] = []
    title = first_non_empty(finding.get("title"), f"Analyse exploratoire {index + 1}")

    story.append(Paragraph(clean_text(title), styles["h2"]))

    summary = finding.get("summary")

    if isinstance(summary, list):
        for item in summary[:2]:
            story.append(Paragraph(clean_text(item), styles["body"]))

    chart = finding_chart(finding)

    if chart is not None:
        story.append(chart)

    story.append(Spacer(1, 7))
    return story


def unresolved_plan_blocks(
    plan: dict[str, Any] | None,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    if not plan:
        return []

    unresolved = [
        request
        for request in list_of_mappings(plan.get("requests"))
        if clean_text(request.get("status")).lower() != "ready"
    ]

    if not unresolved:
        return []

    story: list[Any] = [Paragraph("Interventions requises", styles["h1"])]

    for request in unresolved:
        status = first_non_empty(request.get("status"), "blocked")
        story.append(status_badge(status, styles))
        story.append(Spacer(1, 3))
        story.append(Paragraph(first_non_empty(request.get("request_text"), "Demande non exécutable"), styles["h2"]))

        blockers = request.get("blockers")

        if isinstance(blockers, list):
            for blocker in blockers[:4]:
                story.append(Paragraph(clean_text(blocker), styles["body"]))

        source = first_non_empty(request.get("source_filename"), "-")
        locator = clean_text(request.get("source_locator"))
        story.append(
            Paragraph(
                f"Source : {source}" + (f" - {locator}" if locator else ""),
                styles["small"],
            )
        )
        story.append(Spacer(1, 7))

    return story


# ============================================================
# DATA QUALITY / PREPARATION SECTION
# ============================================================

def quality_issue_badge(
    severity: str,
    styles: dict[str, ParagraphStyle],
) -> Table:
    normalized = clean_text(
        severity
    ).lower()


    if normalized == "important":
        background = colors.HexColor(
            "#FBEAEC"
        )

        text_color = DANGER

        label = "IMPORTANT"

    elif normalized == "moderate":
        background = colors.HexColor(
            "#FFF4DF"
        )

        text_color = WARNING

        label = "MODERE"

    else:
        background = ACCENT_LIGHT

        text_color = ACCENT

        label = "MINEUR"


    style = ParagraphStyle(
        "QualityIssueBadge",
        parent=
            styles[
                "small"
            ],
        fontName=
            "Helvetica-Bold",
        fontSize=
            7.2,
        leading=
            8.8,
        textColor=
            text_color,
    )


    table = Table(
        [
            [
                Paragraph(
                    label,
                    style,
                )
            ]
        ],
        hAlign=
            "LEFT",
    )


    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    background,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    background,
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
            ]
        )
    )


    return table


def quality_preparation_blocks(
    quality_report: (
        dict[
            str,
            Any,
        ]
        | None
    ),
    styles: dict[
        str,
        ParagraphStyle,
    ],
) -> list[Any]:
    if not quality_report:
        return []


    issues = list_of_mappings(
        quality_report.get(
            "issues"
        )
    )


    datasets = list_of_mappings(
        quality_report.get(
            "datasets"
        )
    )


    missing_cells = sum(
        int(
            dataset.get(
                "missing_cell_count",
                0,
            )
            or 0
        )

        for dataset
        in datasets
    )


    duplicate_rows = sum(
        int(
            dataset.get(
                "duplicate_row_count",
                0,
            )
            or 0
        )

        for dataset
        in datasets
    )


    deterministic_proposals = sum(
        1

        for issue
        in issues

        if bool(
            (
                to_mapping(
                    issue.get(
                        "proposal"
                    )
                )
                or {}
            ).get(
                "automatic_safe"
            )
        )
    )


    confirmation_required = sum(
        1

        for issue
        in issues

        if bool(
            (
                to_mapping(
                    issue.get(
                        "proposal"
                    )
                )
                or {}
            ).get(
                "requires_user_confirmation"
            )
        )
    )


    semantic_review_count = int(
        quality_report.get(
            "semantic_review_count",
            0,
        )
        or 0
    )


    story: list[Any] = []


    story.append(
        Paragraph(
            "Qualité et préparation des données",
            styles[
                "h1"
            ],
        )
    )


    story.append(
        Paragraph(
            (
                "Cette section documente les problèmes "
                "détectés avant l'analyse et distingue "
                "ce qui a été détecté, proposé et "
                "réellement appliqué."
            ),
            styles[
                "subtitle"
            ],
        )
    )


    overview_metrics = [
        (
            "Datasets analysés",
            quality_report.get(
                "dataset_count",
                len(
                    datasets
                ),
            ),
        ),
        (
            "Lignes",
            quality_report.get(
                "total_rows",
                0,
            ),
        ),
        (
            "Colonnes",
            quality_report.get(
                "total_columns",
                0,
            ),
        ),
        (
            "Problèmes détectés",
            quality_report.get(
                "issue_count",
                len(
                    issues
                ),
            ),
        ),
        (
            "Cellules manquantes",
            missing_cells,
        ),
        (
            "Doublons stricts",
            duplicate_rows,
        ),
    ]


    story.append(
        metric_table(
            overview_metrics,
            styles,
        )
    )


    story.append(
        Spacer(
            1,
            9,
        )
    )


    story.append(
        Paragraph(
            "Gravité des problèmes",
            styles[
                "h2"
            ],
        )
    )


    story.append(
        metric_table(
            [
                (
                    "Importants",
                    quality_report.get(
                        "important_count",
                        0,
                    ),
                ),
                (
                    "Modérés",
                    quality_report.get(
                        "moderate_count",
                        0,
                    ),
                ),
                (
                    "Mineurs",
                    quality_report.get(
                        "minor_count",
                        0,
                    ),
                ),
                (
                    "Lecture sémantique",
                    semantic_review_count,
                ),
            ],
            styles,
        )
    )


    story.append(
        Spacer(
            1,
            9,
        )
    )


    story.append(
        Paragraph(
            "État des transformations",
            styles[
                "h2"
            ],
        )
    )


    state_rows = [
        [
            Paragraph(
                "DÉTECTÉ",
                styles[
                    "label"
                ],
            ),
            Paragraph(
                (
                    f"{format_number(quality_report.get('issue_count', len(issues)))} "
                    "anomalie(s) documentée(s) par le moteur Python."
                ),
                styles[
                    "body"
                ],
            ),
        ],
        [
            Paragraph(
                "PROPOSÉ",
                styles[
                    "label"
                ],
            ),
            Paragraph(
                (
                    f"{format_number(deterministic_proposals)} "
                    "correction(s) déterministe(s) possible(s). "
                    f"{format_number(confirmation_required)} "
                    "action(s) exigent une validation utilisateur."
                ),
                styles[
                    "body"
                ],
            ),
        ],
        [
            Paragraph(
                "APPLIQUÉ",
                styles[
                    "label"
                ],
            ),
            Paragraph(
                (
                    "0 transformation. Le pipeline de nettoyage "
                    "n'a encore appliqué aucune modification aux "
                    "données sources."
                ),
                styles[
                    "body"
                ],
            ),
        ],
    ]


    state_table = Table(
        state_rows,
        colWidths=[
            29 * mm,
            135 * mm,
        ],
        hAlign=
            "LEFT",
    )


    state_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    PANEL,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    GRID,
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.25,
                    GRID,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )


    story.append(
        state_table
    )


    if issues:
        story.append(
            Spacer(
                1,
                10,
            )
        )


        story.append(
            Paragraph(
                "Principaux problèmes détectés",
                styles[
                    "h2"
                ],
            )
        )


        for issue in issues[:10]:
            evidence = (
                to_mapping(
                    issue.get(
                        "evidence"
                    )
                )
                or {}
            )


            proposal = (
                to_mapping(
                    issue.get(
                        "proposal"
                    )
                )
                or {}
            )


            dataset_name = first_non_empty(
                issue.get(
                    "dataset_filename"
                ),
                "-",
            )


            column = clean_text(
                issue.get(
                    "column"
                )
            )


            location = dataset_name

            if column:
                location = (
                    f"{dataset_name} - "
                    f"{column}"
                )


            title = first_non_empty(
                issue.get(
                    "title"
                ),
                issue.get(
                    "kind"
                ),
                "Problème qualité",
            )


            issue_block: list[Any] = [
                quality_issue_badge(
                    clean_text(
                        issue.get(
                            "severity"
                        )
                    ),
                    styles,
                ),
                Spacer(
                    1,
                    3,
                ),
                Paragraph(
                    clean_text(
                        title
                    ),
                    styles[
                        "h2"
                    ],
                ),
                Paragraph(
                    location,
                    styles[
                        "small"
                    ],
                ),
            ]


            explanation = clean_text(
                issue.get(
                    "explanation"
                )
            )


            if explanation:
                issue_block.append(
                    Paragraph(
                        explanation,
                        styles[
                            "body"
                        ],
                    )
                )


            observed_count = (
                evidence.get(
                    "observed_count"
                )
            )


            affected_ratio = (
                evidence.get(
                    "affected_ratio"
                )
            )


            evidence_parts: list[
                str
            ] = []


            if observed_count is not None:
                evidence_parts.append(
                    (
                        "Observations concernées : "
                        f"{format_number(observed_count)}"
                    )
                )


            if (
                safe_number(
                    affected_ratio
                )
                is not None
            ):
                evidence_parts.append(
                    (
                        "Part concernée : "
                        f"{format_percent(affected_ratio)}"
                    )
                )


            if evidence_parts:
                issue_block.append(
                    Paragraph(
                        " - ".join(
                            evidence_parts
                        ),
                        styles[
                            "small"
                        ],
                    )
                )


            examples = evidence.get(
                "examples"
            )


            if (
                isinstance(
                    examples,
                    list,
                )
                and
                examples
            ):
                example_text = " ; ".join(
                    truncate(
                        example,
                        70,
                    )

                    for example
                    in examples[:4]
                )


                issue_block.append(
                    Paragraph(
                        (
                            "Exemples : "
                            f"{clean_text(example_text)}"
                        ),
                        styles[
                            "quote"
                        ],
                    )
                )


            proposal_description = clean_text(
                proposal.get(
                    "description"
                )
            )


            if proposal_description:
                proposal_state = (
                    "déterministe possible"
                    if bool(
                        proposal.get(
                            "automatic_safe"
                        )
                    )
                    else
                    "décision automatique interdite"
                )


                semantic_note = (
                    " - lecture sémantique recommandée"
                    if bool(
                        issue.get(
                            "semantic_review_recommended"
                        )
                    )
                    else
                    ""
                )


                issue_block.append(
                    Paragraph(
                        (
                            "Action proposée : "
                            f"{proposal_description}"
                        ),
                        styles[
                            "body"
                        ],
                    )
                )


                issue_block.append(
                    Paragraph(
                        (
                            "Statut : "
                            f"{proposal_state}"
                            f"{semantic_note}."
                        ),
                        styles[
                            "small"
                        ],
                    )
                )


            issue_block.append(
                Spacer(
                    1,
                    7,
                )
            )


            story.append(
                KeepTogether(
                    issue_block
                )
            )


        if len(
            issues
        ) > 10:
            story.append(
                Paragraph(
                    (
                        f"{len(issues) - 10} autre(s) "
                        "problème(s) sont présents dans "
                        "le diagnostic complet de DataLens."
                    ),
                    styles[
                        "small"
                    ],
                )
            )


    story.append(
        Spacer(
            1,
            8,
        )
    )


    story.append(
        Paragraph(
            (
                "Traçabilité : moteur "
                f"{first_non_empty(quality_report.get('rule_version'), '-')}. "
                "Modifications silencieuses : 0. "
                "Les données brutes sont conservées."
            ),
            styles[
                "small"
            ],
        )
    )


    story.append(
        Paragraph(
            (
                "Une proposition de nettoyage n'est pas "
                "considérée comme appliquée tant qu'une "
                "opération déterministe n'a pas été validée "
                "et exécutée."
            ),
            styles[
                "small"
            ],
        )
    )


    return story


# ============================================================
# PUBLIC API
# ============================================================

def build_export_filename(title: str | None = None) -> str:
    date_part = datetime.now().astimezone().strftime("%Y-%m-%d")
    base = clean_text(title or "analyse-datalens").lower()
    base = "".join(character if character.isalnum() else "-" for character in base)
    base = "-".join(part for part in base.split("-") if part)
    base = base[:48] or "analyse-datalens"
    return f"datalens-{base}-{date_part}.pdf"


def build_analysis_pdf(
    *,
    report: object,
    objective: str | None = None,
    document_summary: object | None = None,
    requested_analysis_plan: object | None = None,
    quality_report: object | None = None,
) -> bytes:
    report_data = to_mapping(report)

    if report_data is None:
        raise TypeError("The analysis report must be a mapping or Pydantic model.")

    document_data = to_mapping(document_summary)
    plan_data = to_mapping(requested_analysis_plan)
    quality_data = to_mapping(quality_report)

    buffer = BytesIO()
    styles = build_styles()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=17 * mm,
        bottomMargin=19 * mm,
        title=clean_text(report_data.get("title") or "Analyse DataLens"),
        author="DataLens",
        subject="Rapport analytique DataLens",
    )

    story: list[Any] = []

    # Cover / identity
    story.append(Paragraph("DATALENS / RAPPORT ANALYTIQUE", styles["label"]))
    story.append(Paragraph(first_non_empty(report_data.get("title"), "Analyse DataLens"), styles["title"]))

    if objective:
        story.append(Paragraph("Objectif", styles["label"]))
        story.append(Paragraph(clean_text(objective), styles["subtitle"]))

    generated_at = datetime.now().astimezone().strftime("%d/%m/%Y %H:%M")
    story.append(Paragraph(f"Généré localement le {generated_at}", styles["small"]))
    story.append(Spacer(1, 8))

    inventory = to_mapping(report_data.get("inventory")) or {}
    requested = list_of_mappings(report_data.get("requested_findings"))

    top_metrics = [
        ("Datasets", inventory.get("dataset_count", len(report_data.get("datasets") or []))),
        ("Analyses découvertes", inventory.get("discovered_analysis_count", "-")),
        ("Analyses exécutées", inventory.get("executed_analysis_count", "-")),
        ("Analyses demandées", len(requested)),
    ]

    story.append(metric_table(top_metrics, styles))
    story.append(Spacer(1, 12))

    # Executive summary
    story.append(Paragraph("Synthèse exécutive", styles["h1"]))
    executive_summary = report_data.get("executive_summary")

    if isinstance(executive_summary, list) and executive_summary:
        for item in executive_summary:
            story.append(Paragraph("• " + clean_text(item), styles["body"]))
    else:
        story.append(Paragraph("Aucune synthèse exécutive n'est disponible.", styles["body"]))

    if document_data:
        story.append(Spacer(1, 5))
        story.append(Paragraph("Documentation utilisée", styles["h2"]))
        doc_metrics = [
            ("Documents", document_data.get("document_count", 0)),
            ("Demandes détectées", document_data.get("analytical_request_count", 0)),
            ("Claims vérifiés", document_data.get("verified_claim_count", 0)),
        ]
        story.append(metric_table(doc_metrics, styles))

    story.extend(unresolved_plan_blocks(plan_data, styles))

    # Data quality / preparation
    if quality_data:
        story.append(PageBreak())
        story.extend(
            quality_preparation_blocks(
                quality_data,
                styles,
            )
        )

    # Requested findings
    if requested:
        story.append(PageBreak())
        story.append(Paragraph("Analyses demandées", styles["h1"]))
        story.append(
            Paragraph(
                "Résultats correspondant aux demandes explicitement détectées dans la documentation métier.",
                styles["subtitle"],
            )
        )

        for index, finding in enumerate(requested):
            block = requested_finding_block(finding, index, styles)
            story.append(KeepTogether(block))

    # Exploratory findings
    main_findings = list_of_mappings(report_data.get("main_findings"))

    if main_findings:
        story.append(PageBreak())
        story.append(Paragraph("Exploration automatique", styles["h1"]))
        story.append(
            Paragraph(
                "Analyses complémentaires découvertes automatiquement par le moteur déterministe.",
                styles["subtitle"],
            )
        )

        for index, finding in enumerate(main_findings[:6]):
            story.append(KeepTogether(exploratory_finding_block(finding, index, styles)))

    # Quality / methodology
    quality = list_of_mappings(report_data.get("quality"))
    methodology = report_data.get("methodology_notes")

    story.append(PageBreak())
    story.append(Paragraph("Qualité et traçabilité", styles["h1"]))

    if quality:
        story.append(Paragraph("Contrôles qualité", styles["h2"]))

        for item in quality[:8]:
            title = first_non_empty(item.get("title"), item.get("dataset"), "Contrôle qualité")
            summary = item.get("summary")
            story.append(Paragraph(clean_text(title), styles["body"]))

            if isinstance(summary, list):
                for line in summary[:2]:
                    story.append(Paragraph("• " + clean_text(line), styles["small"]))

    if isinstance(methodology, list) and methodology:
        story.append(Paragraph("Méthodologie", styles["h2"]))

        for note in methodology[:16]:
            story.append(Paragraph("• " + clean_text(note), styles["small"]))

    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            f"Version export PDF : {PDF_EXPORT_RULE_VERSION}. Les données brutes ne sont pas incluses dans ce rapport.",
            styles["small"],
        )
    )

    document.build(
        story,
        onFirstPage=page_decorator,
        onLaterPages=page_decorator,
    )

    return buffer.getvalue()
