from __future__ import annotations

from html import (
    escape,
)

from io import (
    BytesIO,
)

from reportlab.lib import (
    colors,
)

from reportlab.lib.enums import (
    TA_CENTER,
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

from app.reporting.schemas import (
    AnalysisReport,
)


PAGE_WIDTH, PAGE_HEIGHT = A4


def pdf_safe_text(
    value: object,
) -> str:
    text = str(
        value
    )

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2022": "-",
        "\u00d7": "x",
        "\u2026": "...",
    }


    for (
        source,
        target,
    ) in replacements.items():
        text = text.replace(
            source,
            target,
        )


    return escape(
        text
    )


def format_number(
    value: int,
) -> str:
    return (
        f"{value:,}"
        .replace(
            ",",
            " ",
        )
    )


def format_bytes(
    value: int,
) -> str:
    if (
        value <
        1024
    ):
        return (
            f"{value} o"
        )


    if (
        value <
        1024 * 1024
    ):
        return (
            f"{value / 1024:.1f} Ko"
        )


    return (
        f"{value / (1024 * 1024):.1f} Mo"
    )


def build_styles():
    base = (
        getSampleStyleSheet()
    )


    title = ParagraphStyle(
        "DataLensTitle",
        parent=
            base["Title"],
        fontName=
            "Helvetica-Bold",
        fontSize=
            24,
        leading=
            29,
        textColor=
            colors.HexColor(
                "#152033"
            ),
        spaceAfter=
            8,
    )


    subtitle = ParagraphStyle(
        "DataLensSubtitle",
        parent=
            base["BodyText"],
        fontName=
            "Helvetica",
        fontSize=
            9,
        leading=
            14,
        textColor=
            colors.HexColor(
                "#64748B"
            ),
        spaceAfter=
            6,
    )


    section_title = ParagraphStyle(
        "DataLensSection",
        parent=
            base["Heading2"],
        fontName=
            "Helvetica-Bold",
        fontSize=
            15,
        leading=
            19,
        textColor=
            colors.HexColor(
                "#172033"
            ),
        spaceBefore=
            8,
        spaceAfter=
            10,
    )


    subsection_title = (
        ParagraphStyle(
            "DataLensSubsection",
            parent=
                base["Heading3"],
            fontName=
                "Helvetica-Bold",
            fontSize=
                11,
            leading=
                15,
            textColor=
                colors.HexColor(
                    "#26344D"
                ),
            spaceAfter=
                6,
        )
    )


    body = ParagraphStyle(
        "DataLensBody",
        parent=
            base["BodyText"],
        fontName=
            "Helvetica",
        fontSize=
            9,
        leading=
            14,
        textColor=
            colors.HexColor(
                "#374151"
            ),
        spaceAfter=
            6,
    )


    small = ParagraphStyle(
        "DataLensSmall",
        parent=
            body,
        fontSize=
            7.5,
        leading=
            11,
        textColor=
            colors.HexColor(
                "#64748B"
            ),
    )


    label = ParagraphStyle(
        "DataLensLabel",
        parent=
            small,
        fontName=
            "Helvetica-Bold",
        textColor=
            colors.HexColor(
                "#4F6B95"
            ),
    )


    centered = ParagraphStyle(
        "DataLensCentered",
        parent=
            small,
        alignment=
            TA_CENTER,
    )


    return {
        "title":
            title,

        "subtitle":
            subtitle,

        "section":
            section_title,

        "subsection":
            subsection_title,

        "body":
            body,

        "small":
            small,

        "label":
            label,

        "centered":
            centered,
    }


def draw_page_footer(
    canvas,
    document,
) -> None:
    canvas.saveState()


    canvas.setStrokeColor(
        colors.HexColor(
            "#E5E7EB"
        )
    )

    canvas.setLineWidth(
        0.5
    )

    canvas.line(
        18 * mm,
        15 * mm,
        PAGE_WIDTH - 18 * mm,
        15 * mm,
    )


    canvas.setFont(
        "Helvetica",
        7,
    )

    canvas.setFillColor(
        colors.HexColor(
            "#7C8798"
        )
    )


    canvas.drawString(
        18 * mm,
        10 * mm,
        "DataLens - rapport genere localement",
    )


    canvas.drawRightString(
        PAGE_WIDTH - 18 * mm,
        10 * mm,
        (
            f"Page "
            f"{document.page}"
        ),
    )


    canvas.restoreState()


def paragraph_list(
    values: list[
        str
    ],
    style,
) -> list:
    result = []


    for value in values:
        result.append(
            Paragraph(
                (
                    "- "
                    +
                    pdf_safe_text(
                        value
                    )
                ),
                style,
            )
        )


    return result


def render_analysis_report_pdf(
    report: AnalysisReport,
) -> bytes:
    buffer = BytesIO()

    document = (
        SimpleDocTemplate(
            buffer,

            pagesize=
                A4,

            rightMargin=
                18 * mm,

            leftMargin=
                18 * mm,

            topMargin=
                18 * mm,

            bottomMargin=
                22 * mm,

            title=
                report.title,

            author=
                "DataLens",
        )
    )


    styles = (
        build_styles()
    )

    story = []


    # ========================================================
    # COVER / INTRODUCTION
    # ========================================================

    story.append(
        Paragraph(
            "DataLens",
            styles[
                "label"
            ],
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    story.append(
        Paragraph(
            pdf_safe_text(
                report.title
            ),
            styles[
                "title"
            ],
        )
    )


    if report.objective:
        story.append(
            Paragraph(
                (
                    "<b>Objectif :</b> "
                    +
                    pdf_safe_text(
                        report.objective
                    )
                ),
                styles[
                    "subtitle"
                ],
            )
        )

    else:
        story.append(
            Paragraph(
                (
                    "Exploration automatique "
                    "des donnees - aucun objectif "
                    "specifique n'a ete fourni."
                ),
                styles[
                    "subtitle"
                ],
            )
        )


    story.append(
        Spacer(
            1,
            7 * mm,
        )
    )


    overview_table = Table(
        [
            [
                Paragraph(
                    "Fichiers",
                    styles[
                        "small"
                    ],
                ),
                Paragraph(
                    "Lignes",
                    styles[
                        "small"
                    ],
                ),
                Paragraph(
                    "Analyses retenues",
                    styles[
                        "small"
                    ],
                ),
                Paragraph(
                    "Relations potentielles",
                    styles[
                        "small"
                    ],
                ),
            ],
            [
                Paragraph(
                    str(
                        report.dataset_count
                    ),
                    styles[
                        "subsection"
                    ],
                ),
                Paragraph(
                    format_number(
                        report.total_rows
                    ),
                    styles[
                        "subsection"
                    ],
                ),
                Paragraph(
                    str(
                        len(
                            report.analyses
                        )
                    ),
                    styles[
                        "subsection"
                    ],
                ),
                Paragraph(
                    str(
                        len(
                            report.relationships
                        )
                    ),
                    styles[
                        "subsection"
                    ],
                ),
            ],
        ],

        colWidths=[
            39 * mm,
            39 * mm,
            51 * mm,
            51 * mm,
        ],
    )


    overview_table.setStyle(
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
                        "#F6F8FB"
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
                    0.5,
                    colors.HexColor(
                        "#DDE3EC"
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
                        "#E6EAF0"
                    ),
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
                    8,
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
                    8,
                ),
            ]
        )
    )


    story.append(
        overview_table
    )

    story.append(
        Spacer(
            1,
            10 * mm,
        )
    )


    # ========================================================
    # EXECUTIVE SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "Synthese",
            styles[
                "section"
            ],
        )
    )

    story.extend(
        paragraph_list(
            report.executive_summary,
            styles[
                "body"
            ],
        )
    )


    # ========================================================
    # DATASETS
    # ========================================================

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    story.append(
        Paragraph(
            "Donnees analysees",
            styles[
                "section"
            ],
        )
    )


    dataset_rows = [
        [
            Paragraph(
                "Fichier",
                styles[
                    "label"
                ],
            ),
            Paragraph(
                "Lignes",
                styles[
                    "label"
                ],
            ),
            Paragraph(
                "Colonnes",
                styles[
                    "label"
                ],
            ),
            Paragraph(
                "Memoire",
                styles[
                    "label"
                ],
            ),
        ]
    ]


    for dataset in (
        report.datasets
    ):
        dataset_rows.append(
            [
                Paragraph(
                    pdf_safe_text(
                        dataset.filename
                    ),
                    styles[
                        "small"
                    ],
                ),

                Paragraph(
                    format_number(
                        dataset.row_count
                    ),
                    styles[
                        "small"
                    ],
                ),

                Paragraph(
                    str(
                        dataset.column_count
                    ),
                    styles[
                        "small"
                    ],
                ),

                Paragraph(
                    format_bytes(
                        dataset.memory_bytes
                    ),
                    styles[
                        "small"
                    ],
                ),
            ]
        )


    dataset_table = Table(
        dataset_rows,

        colWidths=[
            100 * mm,
            25 * mm,
            25 * mm,
            30 * mm,
        ],

        repeatRows=
            1,
    )


    dataset_table.setStyle(
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
                    colors.HexColor(
                        "#EEF3FA"
                    ),
                ),

                (
                    "GRID",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    0.35,
                    colors.HexColor(
                        "#DFE5ED"
                    ),
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
        dataset_table
    )


    # ========================================================
    # ANALYSES
    # ========================================================

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "Analyses proposees",
            styles[
                "section"
            ],
        )
    )

    story.append(
        Paragraph(
            (
                "Les analyses ci-dessous sont "
                "classees par priorite. "
                "Certaines sont deja executables ; "
                "d'autres necessitent encore un "
                "moteur d'execution specialise."
            ),
            styles[
                "body"
            ],
        )
    )


    for (
        index,
        analysis,
    ) in enumerate(
        report.analyses,
        start=1,
    ):
        variable_text = ", ".join(
            (
                f"{variable.column} "
                f"({variable.role})"
            )
            for variable
            in analysis.variables
        )


        analysis_content = [
            Paragraph(
                (
                    f"{index}. "
                    +
                    pdf_safe_text(
                        analysis.title
                    )
                ),
                styles[
                    "subsection"
                ],
            ),

            Table(
                [
                    [
                        Paragraph(
                            "<b>Fichier</b>",
                            styles[
                                "small"
                            ],
                        ),
                        Paragraph(
                            pdf_safe_text(
                                analysis
                                .dataset_filename
                            ),
                            styles[
                                "small"
                            ],
                        ),
                    ],

                    [
                        Paragraph(
                            "<b>Famille</b>",
                            styles[
                                "small"
                            ],
                        ),
                        Paragraph(
                            pdf_safe_text(
                                analysis.family
                            ),
                            styles[
                                "small"
                            ],
                        ),
                    ],

                    [
                        Paragraph(
                            "<b>Priorite</b>",
                            styles[
                                "small"
                            ],
                        ),
                        Paragraph(
                            str(
                                analysis
                                .priority_score
                            ),
                            styles[
                                "small"
                            ],
                        ),
                    ],

                    [
                        Paragraph(
                            "<b>Etat</b>",
                            styles[
                                "small"
                            ],
                        ),
                        Paragraph(
                            pdf_safe_text(
                                analysis.readiness
                            ),
                            styles[
                                "small"
                            ],
                        ),
                    ],

                    [
                        Paragraph(
                            "<b>Graphique</b>",
                            styles[
                                "small"
                            ],
                        ),
                        Paragraph(
                            pdf_safe_text(
                                analysis.chart_type
                            ),
                            styles[
                                "small"
                            ],
                        ),
                    ],

                    [
                        Paragraph(
                            "<b>Variables</b>",
                            styles[
                                "small"
                            ],
                        ),
                        Paragraph(
                            pdf_safe_text(
                                variable_text
                            ),
                            styles[
                                "small"
                            ],
                        ),
                    ],
                ],

                colWidths=[
                    31 * mm,
                    149 * mm,
                ],
            ),
        ]


        if analysis.reasons:
            analysis_content.append(
                Spacer(
                    1,
                    2 * mm,
                )
            )

            analysis_content.append(
                Paragraph(
                    "Pourquoi cette analyse ?",
                    styles[
                        "label"
                    ],
                )
            )

            analysis_content.extend(
                paragraph_list(
                    analysis.reasons,
                    styles[
                        "small"
                    ],
                )
            )


        if analysis.limitations:
            analysis_content.append(
                Paragraph(
                    "Points d'attention",
                    styles[
                        "label"
                    ],
                )
            )

            analysis_content.extend(
                paragraph_list(
                    analysis.limitations,
                    styles[
                        "small"
                    ],
                )
            )


        story.append(
            KeepTogether(
                analysis_content
            )
        )

        story.append(
            Spacer(
                1,
                5 * mm,
            )
        )


    # ========================================================
    # RELATIONSHIPS
    # ========================================================

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "Relations entre fichiers",
            styles[
                "section"
            ],
        )
    )


    if report.relationships:
        for relationship in (
            report.relationships
        ):
            filenames = (
                " / ".join(
                    relationship
                    .dataset_filenames
                )
            )

            columns = (
                ", ".join(
                    relationship
                    .shared_columns
                )
            )


            story.append(
                Paragraph(
                    pdf_safe_text(
                        filenames
                    ),
                    styles[
                        "subsection"
                    ],
                )
            )

            story.append(
                Paragraph(
                    (
                        "<b>Colonnes communes :</b> "
                        +
                        pdf_safe_text(
                            columns
                        )
                    ),
                    styles[
                        "body"
                    ],
                )
            )

            story.append(
                Paragraph(
                    pdf_safe_text(
                        relationship.reason
                    ),
                    styles[
                        "small"
                    ],
                )
            )

            story.append(
                Spacer(
                    1,
                    4 * mm,
                )
            )

    else:
        story.append(
            Paragraph(
                (
                    "Aucune relation potentielle "
                    "entre fichiers n'a ete "
                    "identifiee."
                ),
                styles[
                    "body"
                ],
            )
        )


    # ========================================================
    # ADDITIONAL DATA
    # ========================================================

    story.append(
        Paragraph(
            "Pour aller plus loin",
            styles[
                "section"
            ],
        )
    )


    if (
        report
        .additional_data_suggestions
    ):
        for suggestion in (
            report
            .additional_data_suggestions
        ):
            story.append(
                Paragraph(
                    pdf_safe_text(
                        suggestion.title
                    ),
                    styles[
                        "subsection"
                    ],
                )
            )

            story.append(
                Paragraph(
                    pdf_safe_text(
                        suggestion.rationale
                    ),
                    styles[
                        "body"
                    ],
                )
            )


            if (
                suggestion
                .example_fields
            ):
                story.append(
                    Paragraph(
                        (
                            "<b>Exemples :</b> "
                            +
                            pdf_safe_text(
                                ", ".join(
                                    suggestion
                                    .example_fields
                                )
                            )
                        ),
                        styles[
                            "small"
                        ],
                    )
                )


            story.append(
                Spacer(
                    1,
                    4 * mm,
                )
            )

    else:
        story.append(
            Paragraph(
                (
                    "Aucune donnee complementaire "
                    "n'est suggeree actuellement."
                ),
                styles[
                    "body"
                ],
            )
        )


    # ========================================================
    # METHODOLOGY
    # ========================================================

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "Methodologie et limites",
            styles[
                "section"
            ],
        )
    )

    story.extend(
        paragraph_list(
            report.methodology_notes,
            styles[
                "body"
            ],
        )
    )


    story.append(
        Spacer(
            1,
            8 * mm,
        )
    )

    story.append(
        Paragraph(
            (
                "Version du rapport : "
                +
                pdf_safe_text(
                    report
                    .report_rule_version
                )
            ),
            styles[
                "small"
            ],
        )
    )


    document.build(
        story,

        onFirstPage=
            draw_page_footer,

        onLaterPages=
            draw_page_footer,
    )


    pdf_bytes = (
        buffer.getvalue()
    )

    buffer.close()


    return pdf_bytes