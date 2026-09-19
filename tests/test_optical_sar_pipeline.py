import os
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from models.data.optical_sar import OpticalSARDataLoader
from models.inference.two_stream_fusion import TwoStreamFusionEngine
from backend.app.tools.fusion_segmenter import OpticalSARFusionTool
from backend.app.schemas.tools import OpticalSARFusionInput

@pytest.fixture
def synthetic_bench_pair(tmp_path):
    """Creates a high-fidelity synthetic benchmark pair with known water and built-up areas."""
    return OpticalSARDataLoader.create_synthetic_fixture(
        output_dir=tmp_path / "fixtures",
        sample_id="test_pair_01",
        height=128,
        width=128,
        pixel_size_m=10.0
    )

def test_optical_sar_known_areas_and_geojson(synthetic_bench_pair):
    """Verifies that optical_sar_fusion tool extracts masks and empirical areas in m2, ha, and %."""
    tool = OpticalSARFusionTool()
    res = tool.run(OpticalSARFusionInput(
        optical_path=synthetic_bench_pair["optical_path"],
        sar_path=synthetic_bench_pair["sar_path"],
        confidence_threshold=0.40,
        ablation_mode="fused"
    ))

    # Total area: 128x128 pixels * 100 m2 = 1,638,400 m2 = 163.84 ha = 1.6384 km2
    assert res.classification_map_path is not None
    assert os.path.exists(res.classification_map_path)
    assert res.water_area_m2 > 100_000.0  # Known water: ~2000 px = 200,000 m2
    assert res.water_area_ha > 10.0
    assert res.built_up_area_m2 > 150_000.0 # Known built: ~2750 px = 275,000 m2
    assert res.built_up_area_ha > 15.0
    assert res.water_percentage > 5.0
    assert res.built_up_percentage > 10.0
    assert res.geojson is not None
    assert len(res.geojson.features) > 0

def test_optical_sar_ablation_modes(synthetic_bench_pair):
    """Verifies that fused, optical_only, and sar_only ablation modes execute cleanly."""
    engine = TwoStreamFusionEngine()
    for mode in ["fused", "optical_only", "sar_only"]:
        out = engine.segment_scene(
            optical_path=synthetic_bench_pair["optical_path"],
            sar_path=synthetic_bench_pair["sar_path"],
            ablation_mode=mode
        )
        assert out["ablation_mode"] == mode
        assert os.path.exists(out["classification_map_path"])
        assert out["water_pixels"] >= 0
        assert out["built_up_pixels"] >= 0

def test_resolution_mismatch_auto_resampling(tmp_path, synthetic_bench_pair):
    """Verifies that differing resolutions between optical and SAR are auto-resampled without crashing."""
    # Create low-res SAR (64x64 at 20m res)
    sar_low_res = tmp_path / "sar_low_res.tif"
    h_low, w_low = 64, 64
    sar_data = np.full((2, h_low, w_low), -14.0, dtype=np.float32)
    sar_data[0, 5:25, 5:25] = -22.0 # Water
    sar_data[0, 35:60, 35:60] = 3.0  # Built-up

    # Transform with 2x larger pixel size
    deg_res = 20.0 / 111320.0
    transform = from_origin(77.5946, 12.9716, deg_res, deg_res)
    with rasterio.open(
        sar_low_res, "w", driver="GTiff", height=h_low, width=w_low, count=2,
        dtype=rasterio.float32, crs="EPSG:4326", transform=transform
    ) as dst:
        dst.write(sar_data)

    tool = OpticalSARFusionTool()
    res = tool.run(OpticalSARFusionInput(
        optical_path=synthetic_bench_pair["optical_path"],
        sar_path=str(sar_low_res),
        ablation_mode="fused"
    ))

    assert res.resampling_info is not None
    assert res.resampling_info["source_shape"] == [64, 64]
    assert res.resampling_info["target_shape"] == [128, 128]
    assert os.path.exists(res.classification_map_path)

def test_panchromatic_and_single_pol_degradation(tmp_path):
    """Verifies that 1-band Panchromatic optical and 1-band single-pol SAR degrade cleanly."""
    pan_path = tmp_path / "pan.tif"
    sar1_path = tmp_path / "sar_single.tif"
    h, w = 60, 60
    transform = from_origin(77.5946, 12.9716, 0.0001, 0.0001)

    with rasterio.open(pan_path, "w", driver="GTiff", height=h, width=w, count=1, dtype=rasterio.float32, crs="EPSG:4326", transform=transform) as dst:
        dst.write(np.random.uniform(0.1, 0.9, (1, h, w)).astype(np.float32))

    with rasterio.open(sar1_path, "w", driver="GTiff", height=h, width=w, count=1, dtype=rasterio.float32, crs="EPSG:4326", transform=transform) as dst:
        dst.write(np.random.uniform(-25.0, 5.0, (1, h, w)).astype(np.float32))

    tool = OpticalSARFusionTool()
    res = tool.run(OpticalSARFusionInput(
        optical_path=str(pan_path),
        sar_path=str(sar1_path),
        ablation_mode="fused"
    ))

    assert res.degradation_mode is not None
    assert "panchromatic" in res.degradation_mode.lower()
    assert "single-pol" in res.degradation_mode.lower()
    assert os.path.exists(res.classification_map_path)
