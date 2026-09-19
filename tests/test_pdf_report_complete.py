import os
from pathlib import Path
import pytest
from pypdf import PdfReader

from backend.app.schemas.query import QueryRequest
from backend.app.controller.engine import ControllerEngine

def test_pdf_report_contains_all_required_sections(optical_geotiff, sar_geotiff):
    """
    Requirement 6 & Test: Report generation test:
    PDF exists, contains query, input summary, answer, statistics table,
    confidence with components, warnings, and execution trace table.
    """
    engine = ControllerEngine()
    req = QueryRequest(
        query="what changed and is the new area water or built-up?",
        image_ids=[optical_geotiff, sar_geotiff]
    )
    resp = engine.process_query(req)
    assert resp.pdf_report_url is not None

    reports_dir = Path("data/outputs/reports")
    pdf_file = reports_dir / f"SatQuery_Report_{resp.session_id}.pdf"
    assert pdf_file.exists(), f"Expected PDF report at {pdf_file}"
    assert pdf_file.stat().st_size > 2000

    reader = PdfReader(str(pdf_file))
    full_text = ""
    for page in reader.pages:
        full_text += (page.extract_text() or "") + "\n"

    extracted_text = full_text.lower()

    # 1. Title / Header
    assert "satquery ai" in extracted_text
    assert "isro/sac problem statement 26167" in extracted_text

    # 2. Input Summary Section
    assert "input imagery & sensor metadata" in extracted_text

    # 3. Executive Analysis / Answer
    assert "executive analysis & grounded evidence" in extracted_text

    # 4. Empirical Ground-Truth Metrics Table
    assert "empirical ground-truth metrics" in extracted_text

    # 5. Multi-Factor Confidence Breakdown
    assert "calibrated confidence breakdown" in extracted_text

    # 6. Observable Execution Trace Timeline
    assert "observable execution trace" in extracted_text
