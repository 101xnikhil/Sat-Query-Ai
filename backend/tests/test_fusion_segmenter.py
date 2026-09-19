import os
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_bounds

from models.inference.fusion_segmenter import OpticalSARFusionEngine
from backend.app.tools.fusion_segmenter import OpticalSARFusionTool
from backend.app.schemas.tools import OpticalSARFusionInput, OpticalSARFusionOutput

def test_tile_slice_generation():
    """Verify tiling generates overlapping slices that completely cover arbitrary raster shapes."""
    # Test case: 250x300 raster with tile_size 128 and stride 96
    h, w = 250, 300
    tile_size = 128
    stride = 96
    slices = OpticalSARFusionEngine.generate_tile_slices(h, w, tile_size=tile_size, stride=stride)

    assert len(slices) > 0

    # Ensure every single coordinate in (h, w) is covered by at least one slice
    coverage = np.zeros((h, w), dtype=int)
    for r_s, c_s in slices:
        assert (r_s.stop - r_s.start) <= tile_size
        assert (c_s.stop - c_s.start) <= tile_size
        coverage[r_s, c_s] += 1

    assert np.all(coverage >= 1), "Some pixels were missed by the tiling grid!"

def test_blending_weights_properties():
    """Verify smooth 2D cosine/Hann window has positive weights and symmetry."""
    th, tw = 64, 64
    weights = OpticalSARFusionEngine.compute_blending_weights(th, tw)
    assert weights.shape == (th, tw)
    assert np.all(weights > 0.0), "Weights must be strictly positive"
    assert weights[th // 2, tw // 2] > weights[0, 0], "Center weight must be higher than corner"
    # Symmetry check
    assert np.isclose(weights[0, 0], weights[0, -1], atol=1e-4)
    assert np.isclose(weights[0, 0], weights[-1, 0], atol=1e-4)

def test_segment_tile_single_band_degradation():
    """Verify segment_tile degrades gracefully when optical input is single-band panchromatic."""
    engine = OpticalSARFusionEngine()
    opt_tile = np.random.uniform(50, 200, (1, 64, 64)).astype(np.float32)
    sar_tile = np.random.uniform(0.1, 10.0, (1, 64, 64)).astype(np.float32)

    prob_stack = engine.segment_tile(opt_tile, sar_tile)
    assert prob_stack.shape == (3, 64, 64)
    # Sum of probabilities across classes should be 1.0 everywhere
    prob_sum = np.sum(prob_stack, axis=0)
    assert np.allclose(prob_sum, 1.0, atol=1e-4)

def test_segment_scene_end_to_end(optical_geotiff, sar_geotiff):
    """Verify full end-to-end scene segmentation on test fixtures."""
    engine = OpticalSARFusionEngine(tile_size=64, overlap=16)
    res = engine.segment_scene(optical_geotiff, sar_geotiff, confidence_threshold=0.4)

    assert "classification_map_path" in res
    assert os.path.exists(res["classification_map_path"])
    assert res["water_percentage"] >= 0.0
    assert res["built_up_percentage"] >= 0.0
    assert res["water_area_km2"] >= 0.0
    assert res["built_up_area_km2"] >= 0.0
    assert res["total_pixels"] == 100 * 100

    # Verify classification GeoTIFF has correct shape and valid classes (0, 1, 2)
    with rasterio.open(res["classification_map_path"]) as src:
        assert src.shape == (100, 100)
        assert src.count == 1
        arr = src.read(1)
        unique_vals = set(np.unique(arr))
        assert unique_vals.issubset({0, 1, 2})

def test_optical_sar_fusion_tool_typed(optical_geotiff, sar_geotiff):
    """Verify typed OpticalSARFusionTool conforms to BaseTool interface and returns valid Pydantic output."""
    tool = OpticalSARFusionTool(tile_size=64, overlap=16)
    assert tool.name == "optical_sar_fusion"

    tool_input = OpticalSARFusionInput(
        optical_path=optical_geotiff,
        sar_path=sar_geotiff,
        target_classes=["water", "built_up"],
        confidence_threshold=0.4
    )
    output = tool.run(tool_input)

    assert isinstance(output, OpticalSARFusionOutput)
    assert os.path.exists(output.classification_map_path)
    assert output.water_percentage >= 0.0
    assert output.built_up_percentage >= 0.0
