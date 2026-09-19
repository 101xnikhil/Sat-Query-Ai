import os
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from backend.app.controller.engine import ControllerEngine
from backend.app.schemas.query import QueryRequest
from backend.app.schemas.common import TaskType
from models.data.optical_sar import OpticalSARDataLoader

@pytest.fixture
def optical_sar_concordant_pair(tmp_path):
    """Generates an Optical+SAR pair with strong mutual water signal (high agreement)."""
    return OpticalSARDataLoader.create_synthetic_fixture(
        output_dir=tmp_path / "concordant",
        sample_id="concordant_01",
        height=100,
        width=100,
        pixel_size_m=10.0
    )

@pytest.fixture
def optical_sar_discordant_pair(tmp_path):
    """
    Generates an Optical+SAR pair where Optical has cloud obstruction over the reservoir
    and a localized false water anomaly elsewhere, creating low spatial agreement (< 0.50).
    """
    sample = OpticalSARDataLoader.create_synthetic_fixture(
        output_dir=tmp_path / "discordant",
        sample_id="discordant_01",
        height=100,
        width=100,
        pixel_size_m=10.0
    )
    with rasterio.open(sample["optical_path"]) as src:
        prof = src.profile.copy()
        arr = src.read()

    # Cloud obstruction over real reservoir (rows 10:50, cols 10:60) -> high reflectance
    arr[:, 10:50, 10:60] = 0.35
    # False optical shadow/anomaly elsewhere (rows 60:80, cols 10:30) with high Green, low NIR
    arr[1, 60:80, 10:30] = 0.35
    arr[3, 60:80, 10:30] = 0.02

    with rasterio.open(sample["optical_path"], "w", **prof) as dst:
        dst.write(arr)
    return sample

def test_cross_tool_agreement_high_concordance(optical_sar_concordant_pair):
    """High agreement yields high confidence score and logs agreement in trace."""
    engine = ControllerEngine()
    req = QueryRequest(
        query="use the optical and SAR images together to identify built-up and water regions",
        image_ids=[optical_sar_concordant_pair["optical_path"], optical_sar_concordant_pair["sar_path"]]
    )
    res = engine.process_query(req)

    assert res.task == TaskType.OPTICAL_SAR_ANALYSIS
    assert "cross_tool_ndwi_agreement" in res.computed_metrics
    agreement = res.computed_metrics["cross_tool_ndwi_agreement"]
    assert agreement >= 0.50
    assert res.confidence_score >= 0.70
    assert "Discrepancy Warning" not in res.answer

def test_cross_tool_agreement_low_warning_and_confidence_penalty(optical_sar_discordant_pair):
    """Low agreement lowers confidence score and attaches explicit warning to answer and trace."""
    engine = ControllerEngine()
    req = QueryRequest(
        query="identify built-up and water regions with optical and sar fusion",
        image_ids=[optical_sar_discordant_pair["optical_path"], optical_sar_discordant_pair["sar_path"]]
    )
    res = engine.process_query(req)

    assert res.task == TaskType.OPTICAL_SAR_ANALYSIS
    assert "cross_tool_ndwi_agreement" in res.computed_metrics
    agreement = res.computed_metrics["cross_tool_ndwi_agreement"]
    assert agreement < 0.50
    # Confidence must be penalized
    assert res.confidence_score < 0.85
    # Warning must appear in answer and in trace actions
    assert "Discrepancy Warning" in res.answer
    assert any("Discrepancy Warning" in a for a in res.trace.actions_taken)
