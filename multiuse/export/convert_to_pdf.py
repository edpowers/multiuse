from pathlib import Path
from typing import Any

import polars as pl
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def rich_to_pdf(
    df: pl.DataFrame,
    output_path: str | Path,
    highlighted_columns: list[str] | None = None,
    title: str | None = None,
    font_size: int = 9,
) -> None:
    """
    Convert DataFrame with rich markup to PDF with terminal-like formatting.

    Args:
        df: Polars DataFrame with highlighted columns
        output_path: Path for output PDF
        highlighted_columns: Columns with rich markup (auto-detects *_highlighted)
        title: Optional PDF title
        font_size: Base font size for content
    """
    output_path = Path(output_path)

    # Auto-detect highlighted columns
    if highlighted_columns is None:
        highlighted_columns = [col for col in df.columns if col.endswith("_highlighted")]

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.5 * inch,
    )

    story: list[Any] = []
    styles = getSampleStyleSheet()

    # Monospace style for terminal-like output
    mono_style = ParagraphStyle(
        "Mono",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=font_size,
        leading=font_size + 2,
        alignment=TA_LEFT,
        leftIndent=0,
        rightIndent=0,
        spaceAfter=0,
        spaceBefore=0,
    )

    if title:
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Heading1"],
            fontSize=14,
            fontName="Courier-Bold",
            spaceAfter=20,
        )
        story.append(Paragraph(title, title_style))

    # Track column order for consistent display
    display_order = highlighted_columns + ["ROW_INDEX_highlighted"]

    # Process each row
    for idx, row_dict in enumerate(df.iter_rows(named=True)):
        if idx > 0:
            # Add divider between records
            story.append(Spacer(1, 6))
            story.append(Paragraph("─" * 80, mono_style))
            story.append(Spacer(1, 6))

        # Display each field in order
        for col in display_order:
            if col in row_dict:
                value = row_dict[col]
                if value is not None and str(value).strip():
                    # Field label
                    label = f"<b>{col}:</b>"
                    story.append(Paragraph(label, mono_style))

                    # Field value with rich markup converted
                    formatted_value = _rich_to_reportlab_inline(str(value))
                    story.append(Paragraph(formatted_value, mono_style))
                    story.append(Spacer(1, 4))

    doc.build(story)


def _rich_to_reportlab_inline(rich_text: str) -> str:
    """
    Convert rich markup to reportlab inline markup preserving exact spacing.
    """
    # Color mappings
    highlights = {
        "yellow": "#FFFF99",
        "bright_red": "#FFB3B3",
    }

    text_colors = {
        "green": "#00AA00",
        "orange1": "#FF8800",
        "cyan": "#00AAAA",
        "bright_cyan": "#00FFFF",
    }

    result = rich_text

    # Process highlights (background colors) - BEFORE escaping
    for color_name, hex_color in highlights.items():
        result = result.replace(f"[{color_name}]", f"<<<SPAN_START_{color_name}>>>")
        result = result.replace(f"[/{color_name}]", f"<<<SPAN_END_{color_name}>>>")

    # Process text colors - BEFORE escaping
    for color_name, hex_color in text_colors.items():
        result = result.replace(f"[{color_name}]", f"<<<FONT_START_{color_name}>>>")
        result = result.replace(f"[/{color_name}]", f"<<<FONT_END_{color_name}>>>")

    # Generic close tag
    result = result.replace("[/]", "<<<CLOSE_TAG>>>")

    # NOW escape XML special characters
    result = result.replace("&", "&amp;")
    result = result.replace("<", "&lt;")
    result = result.replace(">", "&gt;")

    # Convert placeholders to actual markup
    for color_name, hex_color in highlights.items():
        result = result.replace(
            f"&lt;&lt;&lt;SPAN_START_{color_name}&gt;&gt;&gt;",
            f'<span backColor="{hex_color}">',
        )
        result = result.replace(f"&lt;&lt;&lt;SPAN_END_{color_name}&gt;&gt;&gt;", "</span>")

    for color_name, hex_color in text_colors.items():
        result = result.replace(
            f"&lt;&lt;&lt;FONT_START_{color_name}&gt;&gt;&gt;",
            f'<font color="{hex_color}">',
        )
        result = result.replace(f"&lt;&lt;&lt;FONT_END_{color_name}&gt;&gt;&gt;", "</font>")

    result = result.replace("&lt;&lt;&lt;CLOSE_TAG&gt;&gt;&gt;", "</span></font>")

    return result


def export_search_results_to_pdf(
    results: pl.DataFrame,
    output_path: str | Path,
    title: str | None = None,
    sample_size: int | None = None,
) -> None:
    """
    Export highlighted DataFrame to PDF.

    Args:
        results: DataFrame with *_highlighted columns already created
        output_path: Output PDF path
        title: Optional title for PDF
        sample_size: If provided, sample this many rows
    """
    if sample_size and len(results) > sample_size:
        results = results.sample(min(sample_size, len(results)))

    rich_to_pdf(
        df=results,
        output_path=output_path,
        title=title,
    )
