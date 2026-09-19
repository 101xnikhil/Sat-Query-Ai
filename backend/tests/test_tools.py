import pytest
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.base import ToolParameterError
from backend.app.schemas.tools import (
    RSVQAOutput, RSCaptionOutput, RSGroundingOutput,
    ChangeVQAOutput, ChangeMapOutput, OpticalSARFusionOutput, SpectralIndexOutput
)

def test_registry_contains_all_tools():
    reg = ToolRegistry.get_instance()
    expected = [
        "rs_vqa", "rs_caption", "rs_grounding",
        "change_vqa", "change_map", "optical_sar_fusion", "spectral_index"
    ]
    for tool_name in expected:
        assert reg.has(tool_name)

def test_tool_whitelisted_parameter_enforcement(optical_geotiff):
    reg = ToolRegistry.get_instance()
    tool = reg.get("rs_vqa")
    
    # Valid parameters
    out = tool.execute({"query": "What is the land use?", "image_path": optical_geotiff})
    assert isinstance(out, RSVQAOutput)

    # Invalid / unwhitelisted parameter
    with pytest.raises(ToolParameterError) as exc_info:
        tool.execute({
            "query": "What is the land use?",
            "image_path": optical_geotiff,
            "unauthorized_secret_param": 123
        })
    assert "not in the whitelist" in str(exc_info.value)

def test_spectral_index_ndvi(optical_geotiff):
    reg = ToolRegistry.get_instance()
    out: SpectralIndexOutput = reg.execute("spectral_index", {
        "image_path": optical_geotiff,
        "index_type": "NDVI",
        "threshold": 0.2
    })
    assert out.index_type == "NDVI"
    assert -1.0 <= out.mean_index <= 1.0
    assert out.positive_percentage >= 0.0

def test_change_map_stub(optical_geotiff):
    reg = ToolRegistry.get_instance()
    out: ChangeMapOutput = reg.execute("change_map", {
        "image_t1_path": optical_geotiff,
        "image_t2_path": optical_geotiff,
        "threshold": 0.5
    })
    assert isinstance(out.area_changed_km2, float)
    assert isinstance(out.percentage_changed, float)
    assert out.direction_of_change != ""

def test_grounding_stub(optical_geotiff):
    reg = ToolRegistry.get_instance()
    out: RSGroundingOutput = reg.execute("rs_grounding", {
        "query": "Locate all ships",
        "image_path": optical_geotiff
    })
    assert out.detected_count > 0
    assert len(out.geojson.features) == out.detected_count
