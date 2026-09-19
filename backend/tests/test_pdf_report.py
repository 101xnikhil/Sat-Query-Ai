import os
from pathlib import Path
import pytest

from backend.app.schemas.query import QueryResponse, LayerItem
from backend.app.schemas.common import TaskType
from backend.app.schemas.trace import ExecutionTrace, ValidationRecord, ToolExecutionRecord
from backend.app.reporting.pdf_exporter import SatellitePDFReportGenerator

def test_pdf_report_generation(test_data_dir):
    """Verify SatellitePDFReportGenerator creates a valid, compliant PDF file."""
    output_dir = os.path.join(test_data_dir, "reports")
    generator = SatellitePDFReportGenerator(output_dir=output_dir)

    dummy_response = QueryResponse(
        session_id="test_satquery_session_001",
        query="Identify built-up and water regions using both Optical and SAR sensors",
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        answer=(
            "Cross-modal Optical+SAR two-stream analysis detected:\n"
            "• Surface Water: 16.9617 km² (13.69% of scene)\n"
            "• Built-up Urban: 9.9044 km² (7.99% of scene)\n"
            "• Spectral Cross-Check: NDWI spatial agreement score is 75.5%."
        ),
        layers=[],
        computed_metrics={
            "water_area_km2": 16.9617,
            "water_percentage": 13.69,
            "built_up_area_km2": 9.9044,
            "built_up_percentage": 7.99,
            "cross_tool_ndwi_agreement": 0.7547
        },
        confidence_score=0.885,
        trace=ExecutionTrace(
            session_id="test_satquery_session_001",
            task=TaskType.OPTICAL_SAR_ANALYSIS,
            timestamp_utc="2026-09-19T12:30:00Z",
            input_validation=ValidationRecord(passed=True, errors=[], warnings=[]),
            tool_calls=[
                ToolExecutionRecord(
                    tool_name="optical_sar_fusion",
                    parameters={"confidence_threshold": 0.5},
                    status="success",
                    duration_ms=42.5
                ),
                ToolExecutionRecord(
                    tool_name="spectral_index",
                    parameters={"index_type": "NDWI", "threshold": 0.15},
                    status="success",
                    duration_ms=15.1
                )
            ],
            computed_metrics={
                "water_area_km2": 16.9617,
                "water_percentage": 13.69,
                "built_up_area_km2": 9.9044,
                "built_up_percentage": 7.99,
                "cross_tool_ndwi_agreement": 0.7547
            },
            confidence_score=0.885,
            confidence_breakdown={
                "fusion_segmenter": 0.90,
                "cross_tool_ndwi_agreement": 0.7547
            },
            total_duration_ms=57.6
        )
    )

    pdf_path = generator.generate_report(dummy_response)

    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000, "PDF file must be non-empty and formatted"

    with open(pdf_path, "rb") as f:
        header = f.read(5)
        assert header == b"%PDF-", "Generated file must have valid PDF magic bytes"
