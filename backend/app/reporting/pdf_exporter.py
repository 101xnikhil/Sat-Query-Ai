import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Image as RLImage
)

from ..schemas.query import QueryResponse
from ..schemas.common import TaskType

def _generate_raster_thumbnail(tif_path: str, output_dir: Path) -> Optional[str]:
    """Generates a high-quality preview PNG thumbnail from a GeoTIFF for inclusion in PDF reports."""
    try:
        import rasterio
        from PIL import Image as PILImage

        p = Path(tif_path)
        if not p.exists():
            return None
        output_dir.mkdir(parents=True, exist_ok=True)
        out_png = output_dir / f"{p.stem}_thumb.png"
        if out_png.exists():
            return str(out_png)

        with rasterio.open(str(p)) as src:
            if src.count >= 3:
                r = src.read(1)
                g = src.read(2)
                b = src.read(3)
                rgb = np.stack([r, g, b], axis=-1).astype(float)
                for i in range(3):
                    c = rgb[:, :, i]
                    mi, ma = np.nanmin(c), np.nanmax(c)
                    if ma > mi:
                        rgb[:, :, i] = (c - mi) / (ma - mi) * 255.0
                    else:
                        rgb[:, :, i] = 0.0
                arr_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
            else:
                arr = src.read(1).astype(float)
                # If boolean or binary mask, scale 0/1 to 0/255
                mi, ma = np.nanmin(arr), np.nanmax(arr)
                if ma > mi:
                    arr_scaled = (arr - mi) / (ma - mi) * 255.0
                else:
                    arr_scaled = arr * 255.0
                arr_uint8 = np.clip(arr_scaled, 0, 255).astype(np.uint8)

        img = PILImage.fromarray(arr_uint8)
        img.thumbnail((260, 260))
        img.save(str(out_png), format="PNG")
        return str(out_png)
    except Exception:
        return None

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
        warn_bg = colors.HexColor("#FEF3C7")
        warn_border = colors.HexColor("#F59E0B")

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=navy_dark
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=12,
            textColor=cyan_accent
        )
        h2_style = ParagraphStyle(
            "SectionH2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=navy_dark,
            spaceBefore=10,
            spaceAfter=5
        )
        body_style = ParagraphStyle(
            "BodyDark",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=text_dark
        )
        callout_style = ParagraphStyle(
            "CalloutText",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#0F172A")
        )
        table_header_style = ParagraphStyle(
            "TableHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white
        )
        table_cell_style = ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
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
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=1.5, color=cyan_accent, spaceBefore=2, spaceAfter=8))

        # 2. Mission Overview & Query
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
                Paragraph(f"<b>Active Layers:</b> {len(response.layers)} layer(s)", body_style)
            ]
        ]
        overview_table = Table(overview_data, colWidths=[360, 180])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), slate_bg),
            ('BOX', (0, 0), (-1, -1), 1, border_color),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 8))

        # 3. Input Summary Table (Files, Modality, Dates, CRS)
        trace = response.trace
        val_rec = trace.input_validation if trace else None
        crs_info = val_rec.crs_check.get("target_crs", "EPSG:4326") if val_rec and val_rec.crs_check else "EPSG:4326"
        
        # Extract files from trace tool call parameters
        input_files = []
        if trace and trace.tool_calls:
            for tc in trace.tool_calls:
                for k, v in tc.parameters.items():
                    if isinstance(v, str) and ("data/" in v or v.endswith(".tif") or v.endswith(".png")):
                        if v not in input_files:
                            input_files.append(v)

        input_summary_rows = [
            [
                Paragraph("<b>Input Image / Asset</b>", table_header_style),
                Paragraph("<b>Modality</b>", table_header_style),
                Paragraph("<b>Coordinate System (CRS)</b>", table_header_style),
                Paragraph("<b>Acquisition Date</b>", table_header_style)
            ]
        ]
        if input_files:
            for f_path in input_files:
                p_name = Path(f_path).name
                mod_str = "SAR (Radar)" if "sar" in p_name.lower() else "Optical / Multispectral"
                date_str = "Metadata Tagged" if "t1" not in p_name else "Observation T1"
                if "t2" in p_name:
                    date_str = "Observation T2"
                input_summary_rows.append([
                    Paragraph(p_name, table_cell_style),
                    Paragraph(mod_str, table_cell_style),
                    Paragraph(crs_info, table_cell_style),
                    Paragraph(date_str, table_cell_style)
                ])
        else:
            input_summary_rows.append([
                Paragraph("Session Target Image", table_cell_style),
                Paragraph("Optical / SAR", table_cell_style),
                Paragraph(crs_info, table_cell_style),
                Paragraph("Acquisition Synchronized", table_cell_style)
            ])

        input_table = Table(input_summary_rows, colWidths=[200, 120, 110, 110])
        input_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), navy_medium),
            ('BOX', (0, 0), (-1, -1), 1, border_color),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, slate_bg])
        ]))
        story.append(Paragraph("1. Input Imagery & Sensor Metadata", h2_style))
        story.append(input_table)
        story.append(Spacer(1, 8))

        # 4. Warnings Callout (if any)
        warnings_list = []
        if val_rec and val_rec.warnings:
            warnings_list.extend(val_rec.warnings)
        if trace and trace.actions_taken:
            for a in trace.actions_taken:
                if "warning" in a.lower() or "discrepancy" in a.lower():
                    warnings_list.append(a)

        if warnings_list:
            warn_p = [Paragraph("<b>Quality Notices & Degradation Warnings:</b>", ParagraphStyle("WTitle", parent=table_cell_style, fontName="Helvetica-Bold", textColor=colors.HexColor("#92400E")))]
            for w in warnings_list[:3]:
                warn_p.append(Paragraph(f"• {w}", ParagraphStyle("WItem", parent=table_cell_style, textColor=colors.HexColor("#78350F"))))
            warn_box = Table([[warn_p]], colWidths=[540])
            warn_box.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), warn_bg),
                ('BOX', (0, 0), (-1, -1), 1, warn_border),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(warn_box)
            story.append(Spacer(1, 8))

        # 5. Executive Analysis / Answer
        story.append(Paragraph("2. Executive Analysis & Grounded Evidence", h2_style))
        answer_text = response.answer.replace("\n", "<br/>")
        answer_box = Table([[Paragraph(answer_text, callout_style)]], colWidths=[540])
        answer_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#3B82F6")),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(answer_box)
        story.append(Spacer(1, 8))

        # 6. Map Figures / Visual Overlays (if generated thumbnails exist)
        figure_elements = []
        thumb_dir = self.output_dir / "thumbnails"
        for f_path in input_files:
            thumb = _generate_raster_thumbnail(f_path, thumb_dir)
            if thumb and os.path.exists(thumb):
                caption = f"Figure: {Path(f_path).name} Preview"
                cell = [
                    RLImage(thumb, width=160, height=160),
                    Spacer(1, 2),
                    Paragraph(caption, ParagraphStyle("FigCap", parent=table_cell_style, alignment=1, fontSize=7))
                ]
                figure_elements.append(cell)
            if len(figure_elements) >= 2:
                break

        if figure_elements:
            fig_table = Table([figure_elements], colWidths=[270] * len(figure_elements))
            fig_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(Paragraph("3. Spatial Overlays & Map Figures", h2_style))
            story.append(fig_table)
            story.append(Spacer(1, 8))

        # 7. Computed Quantitative Metrics Table
        story.append(Paragraph("4. Empirical Ground-Truth Metrics", h2_style))
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
                if k in ["errors", "validation_passed", "resampling_info", "per_class_change"]:
                    continue
                k_clean = k.replace("_", " ").title()
                if isinstance(v, float):
                    v_str = f"{v:.4f}" if abs(v) < 10.0 else f"{v:,.2f}"
                else:
                    v_str = str(v)
                
                unit = "Count / State"
                if "m2" in k:
                    unit = "Square Meters (m²)"
                elif "ha" in k:
                    unit = "Hectares (ha)"
                elif "km2" in k:
                    unit = "Square Kilometers (km²)"
                elif "percentage" in k or "pct" in k:
                    unit = "Percentage (%) of Scene"
                elif "agreement" in k:
                    unit = "Spatial IoU Overlap [0 - 1]"
                elif "direction" in k:
                    unit = "Classified Spectral Vector"

                metric_rows.append([
                    Paragraph(k_clean, table_cell_style),
                    Paragraph(f"<b>{v_str}</b>", table_cell_style),
                    Paragraph(unit, table_cell_style)
                ])

            metrics_table = Table(metric_rows, colWidths=[190, 130, 220])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), navy_medium),
                ('BOX', (0, 0), (-1, -1), 1, border_color),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, slate_bg])
            ]))
            story.append(metrics_table)
        else:
            story.append(Paragraph("No quantitative metrics generated for this task.", table_cell_style))

        story.append(Spacer(1, 8))

        # 8. Calibrated Confidence Breakdown
        if trace and trace.confidence_breakdown:
            story.append(Paragraph("5. Multi-Factor Calibrated Confidence Breakdown", h2_style))
            conf_rows = [
                [
                    Paragraph("<b>Confidence Factor</b>", table_header_style),
                    Paragraph("<b>Sub-Score</b>", table_header_style),
                    Paragraph("<b>Description</b>", table_header_style)
                ]
            ]
            factor_descs = {
                "overall": "Final multi-factor calibrated reliability score",
                "token_probability": "Model logit probability / baseline token confidence",
                "cross_tool_agreement": "Spatial consensus between complementary tools (e.g. Fusion vs NDWI)",
                "input_quality": "Sensor quality penalty score (1.0 minus degradation penalties)",
                "resampling_penalty": "Deduction for spatial grid interpolation / reprojection",
                "missing_bands_penalty": "Deduction for missing multispectral channels",
                "low_overlap_penalty": "Deduction for spatial footprint overlap below 50%"
            }
            for factor_key, factor_val in trace.confidence_breakdown.items():
                k_clean = factor_key.replace("_", " ").title()
                val_str = f"{factor_val * 100:.1f}%" if isinstance(factor_val, float) else str(factor_val)
                desc = factor_descs.get(factor_key, "Domain-specific verification factor")
                conf_rows.append([
                    Paragraph(k_clean, table_cell_style),
                    Paragraph(f"<b>{val_str}</b>", table_cell_style),
                    Paragraph(desc, table_cell_style)
                ])

            conf_table = Table(conf_rows, colWidths=[170, 100, 270])
            conf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), navy_medium),
                ('BOX', (0, 0), (-1, -1), 1, border_color),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, slate_bg])
            ]))
            story.append(conf_table)
            story.append(Spacer(1, 8))

        # 9. Execution Trace Audit Timeline
        if trace and trace.tool_calls:
            story.append(Paragraph("6. Observable Execution Trace & Tool Provenance", h2_style))
            fb_text = "<b>Fallback Router:</b> ACTIVATED" if getattr(trace, "fallback_used", False) else "<b>Planner Mode:</b> Instruction LLM Structured Plan"
            story.append(Paragraph(f"<i>Execution Provenance &bull; {fb_text} &bull; Total Duration: {trace.total_duration_ms:.1f} ms</i>", table_cell_style))
            story.append(Spacer(1, 3))

            trace_rows = [
                [
                    Paragraph("<b>Step #</b>", table_header_style),
                    Paragraph("<b>Tool Name</b>", table_header_style),
                    Paragraph("<b>Latency</b>", table_header_style),
                    Paragraph("<b>Status</b>", table_header_style),
                    Paragraph("<b>Permitted Parameters</b>", table_header_style)
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

            trace_table = Table(trace_rows, colWidths=[35, 115, 60, 60, 270])
            trace_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), navy_medium),
                ('BOX', (0, 0), (-1, -1), 1, border_color),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, slate_bg])
            ]))
            story.append(trace_table)

        # 10. Footer
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=0.75, color=border_color, spaceBefore=2, spaceAfter=4))
        footer_text = (
            "SatQuery AI &bull; ISRO/SAC Problem Statement 26167 &bull; Autonomous Vision-Language Assistant &bull; "
            "Evidence Grounded & Verified Provenance"
        )
        story.append(Paragraph(footer_text, ParagraphStyle("Footer", parent=table_cell_style, alignment=1, textColor=colors.gray)))

        doc.build(story)
        return pdf_path
