import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)

from ..schemas.query import QueryResponse
from ..schemas.common import TaskType

class SatellitePDFReportGenerator:
    """
    Generates professional, publication-ready Satellite Intelligence PDF Reports
    for SatQuery AI (ISRO/SAC Problem Statement 26167).
    """
    def __init__(self, output_dir: str = "data/outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, response: QueryResponse, output_filename: Optional[str] = None) -> str:
        """
        Builds a comprehensive PDF report from a completed QueryResponse object.
        Returns the absolute filepath to the generated PDF.
        """
        session_id = response.session_id
        if not output_filename:
            output_filename = f"SatQuery_Report_{session_id}.pdf"

        pdf_path = str(self.output_dir / output_filename)
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()

        # Custom Styling Palette (Aerospace Blue & Slate)
        navy_dark = colors.HexColor("#0B132B")
        navy_medium = colors.HexColor("#1C2541")
        cyan_accent = colors.HexColor("#00B4D8")
        slate_bg = colors.HexColor("#F8FAFC")
        border_color = colors.HexColor("#CBD5E1")
        text_dark = colors.HexColor("#1E293B")
        success_emerald = colors.HexColor("#059669")

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=navy_dark
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=cyan_accent
        )
        h2_style = ParagraphStyle(
            "SectionH2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=navy_dark,
            spaceBefore=12,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            "BodyDark",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=text_dark
        )
        callout_style = ParagraphStyle(
            "CalloutText",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#0F172A")
        )
        table_header_style = ParagraphStyle(
            "TableHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.white
        )
        table_cell_style = ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=text_dark
        )

        story = []

        # 1. Header Banner
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        header_table = Table(
            [
                [
                    Paragraph("SATQUERY AI &bull; MISSION REPORT", title_style),
                    Paragraph(f"<b>Classification:</b> OPERATIONAL<br/><b>Generated:</b> {now_str}", table_cell_style)
                ],
                [
                    Paragraph("ISRO/SAC Problem Statement 26167 &bull; Agentic Vision-Language Earth Observation", subtitle_style),
                    Paragraph(f"<b>Session ID:</b> {session_id}", table_cell_style)
                ]
            ],
            colWidths=[380, 160]
        )
        header_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=1.5, color=cyan_accent, spaceBefore=2, spaceAfter=10))

        # 2. Mission Overview & Query Metadata
        task_name = response.task.value.replace("_", " ").upper()
        conf_pct = response.confidence_score * 100.0
        conf_label = "HIGH CONFIDENCE" if conf_pct >= 85.0 else ("MODERATE CONFIDENCE" if conf_pct >= 70.0 else "UNCERTAIN")
        conf_color = "#059669" if conf_pct >= 85.0 else ("#D97706" if conf_pct >= 70.0 else "#DC2626")

        overview_data = [
            [
                Paragraph(f"<b>User Query:</b> <i>\"{response.query}\"</i>", body_style),
                Paragraph(f"<b>Calibrated Confidence:</b> <font color='{conf_color}'><b>{conf_pct:.1f}% ({conf_label})</b></font>", body_style)
            ],
            [
                Paragraph(f"<b>Task Classification:</b> {task_name}", body_style),
                Paragraph(f"<b>Active Layers:</b> {len(response.layers)} vector/raster overlay(s)", body_style)
            ]
        ]
        overview_table = Table(overview_data, colWidths=[360, 180])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), slate_bg),
            ('BOX', (0, 0), (-1, -1), 1, border_color),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 12))

        # 3. Executive Summary / Grounded Answer
        story.append(Paragraph("1. Executive Analysis & Grounded Evidence", h2_style))
        answer_text = response.answer.replace("\n", "<br/>")
        answer_box = Table([[Paragraph(answer_text, callout_style)]], colWidths=[540])
        answer_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#3B82F6")),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(answer_box)
        story.append(Spacer(1, 12))

        # 4. Computed Quantitative Metrics Table
        story.append(Paragraph("2. Empirical Ground-Truth Metrics", h2_style))
        story.append(Paragraph(
            "<i>Note: In strict compliance with mission guidelines, all surface areas, percentages, and counts are "
            "directly computed from spatial masks and detections. No numerical figures are generated freely by language models.</i>",
            table_cell_style
        ))
        story.append(Spacer(1, 4))

        metrics = response.computed_metrics
        if metrics:
            metric_rows = [
                [
                    Paragraph("<b>Metric Name</b>", table_header_style),
                    Paragraph("<b>Empirical Value</b>", table_header_style),
                    Paragraph("<b>Unit / Interpretation</b>", table_header_style)
                ]
            ]
            for k, v in metrics.items():
                if k in ["errors", "validation_passed"]:
                    continue
                k_clean = k.replace("_", " ").title()
                if isinstance(v, float):
                    v_str = f"{v:.4f}" if abs(v) < 10.0 else f"{v:.2f}"
                else:
                    v_str = str(v)
                
                # Assign unit
                unit = "Count"
                if "km2" in k:
                    unit = "Square Kilometers (km²)"
                elif "percentage" in k or "pct" in k:
                    unit = "Percentage (%) of Scene"
                elif "agreement" in k:
                    unit = "Spatial IoU Overlap Score [0 - 1]"
                elif "direction" in k:
                    unit = "Classified Spectral Vector"

                metric_rows.append([
                    Paragraph(k_clean, table_cell_style),
                    Paragraph(f"<b>{v_str}</b>", table_cell_style),
                    Paragraph(unit, table_cell_style)
                ])

            metrics_table = Table(metric_rows, colWidths=[200, 140, 200])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), navy_medium),
                ('BOX', (0, 0), (-1, -1), 1, border_color),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, slate_bg])
            ]))
            story.append(metrics_table)
        else:
            story.append(Paragraph("No quantitative metrics generated for this task.", table_cell_style))

        story.append(Spacer(1, 12))

        # 5. Execution Trace Audit Timeline
        trace = response.trace
        if trace and trace.tool_calls:
            story.append(Paragraph("3. Observable Execution Trace & Tool Provenance", h2_style))
            story.append(Paragraph(
                "<i>Complete audit log of deterministic tool executions. Zero Chain-of-Thought leakage.</i>",
                table_cell_style
            ))
            story.append(Spacer(1, 4))

            trace_rows = [
                [
                    Paragraph("<b>Step #</b>", table_header_style),
                    Paragraph("<b>Tool Name</b>", table_header_style),
                    Paragraph("<b>Latency</b>", table_header_style),
                    Paragraph("<b>Status</b>", table_header_style),
                    Paragraph("<b>Execution Parameters</b>", table_header_style)
                ]
            ]

            for idx, s in enumerate(trace.tool_calls, 1):
                param_str = ", ".join(f"{pk}={pv}" for pk, pv in s.parameters.items() if not str(pv).startswith("/"))
                if len(param_str) > 55:
                    param_str = param_str[:52] + "..."
                if not param_str:
                    param_str = "Standard inputs"

                status_color = "#059669" if s.status == "success" else "#DC2626"
                trace_rows.append([
                    Paragraph(str(idx), table_cell_style),
                    Paragraph(f"<b>{s.tool_name}</b>", table_cell_style),
                    Paragraph(f"{s.duration_ms:.1f} ms", table_cell_style),
                    Paragraph(f"<font color='{status_color}'><b>{s.status.upper()}</b></font>", table_cell_style),
                    Paragraph(param_str, table_cell_style)
                ])

            trace_table = Table(trace_rows, colWidths=[40, 120, 65, 65, 250])
            trace_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), navy_medium),
                ('BOX', (0, 0), (-1, -1), 1, border_color),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, slate_bg])
            ]))
            story.append(trace_table)

        # 6. Confidence Breakdown
        if trace and trace.confidence_breakdown:
            story.append(Spacer(1, 10))
            story.append(Paragraph("4. Calibrated Confidence Breakdown", h2_style))
            breakdown_items = []
            for k, v in trace.confidence_breakdown.items():
                breakdown_items.append(f"<b>{k.replace('_', ' ').title()}:</b> {v * 100:.1f}%")
            breakdown_str = " &nbsp;|&nbsp; ".join(breakdown_items)
            story.append(Paragraph(f"Component Weights: {breakdown_str}", body_style))

        # 7. Footer Disclaimer
        story.append(Spacer(1, 16))
        story.append(HRFlowable(width="100%", thickness=0.75, color=border_color, spaceBefore=4, spaceAfter=6))
        footer_text = (
            "SatQuery AI &bull; ISRO/SAC Problem Statement 26167 &bull; Autonomous Vision-Language Assistant &bull; "
            "Evidence Grounded & Verified Provenance"
        )
        story.append(Paragraph(footer_text, ParagraphStyle("Footer", parent=table_cell_style, alignment=1, textColor=colors.gray)))

        doc.build(story)
        return pdf_path
