import pytest
from backend.tools.registry import ToolRegistry
from backend.app.tools.base import ToolParameterError

ALL_SEVEN_TOOLS = [
    "rs_vqa",
    "rs_caption",
    "rs_grounding",
    "change_vqa",
    "change_map",
    "optical_sar_fusion",
    "spectral_index"
]

@pytest.fixture
def registry():
    return ToolRegistry.get_instance()

def test_all_seven_tools_registered(registry):
    for tool_name in ALL_SEVEN_TOOLS:
        assert registry.has(tool_name)
        tool = registry.get(tool_name)
        assert len(tool.whitelisted_params) > 0

@pytest.mark.parametrize("tool_name", ALL_SEVEN_TOOLS)
def test_whitelist_enforcement_rejects_unwhitelisted_parameter(registry, tool_name, optical_geotiff):
    tool = registry.get(tool_name)
    bad_params = {
        "unauthorized_extra_param": "malicious_or_unknown_value",
        "random_debug_flag": True
    }
    with pytest.raises(ToolParameterError, match="not in the whitelist"):
        tool.execute(bad_params)

def test_tool_stubs_return_schema_valid_outputs_labelled_stub(registry, optical_geotiff, sar_geotiff):
    # 1. rs_vqa
    vqa_out = registry.execute("rs_vqa", {
        "query": "Is there a water body?",
        "image_path": optical_geotiff,
        "confidence_threshold": 0.5
    })
    assert hasattr(vqa_out, "answer")
    assert hasattr(vqa_out, "mode")
    assert vqa_out.mode == "stub"

    # 2. rs_caption
    cap_out = registry.execute("rs_caption", {
        "image_path": optical_geotiff,
        "max_length": 100,
        "style": "detailed"
    })
    assert hasattr(cap_out, "caption")
    assert cap_out.mode == "stub"

    # 3. rs_grounding
    grd_out = registry.execute("rs_grounding", {
        "query": "Locate water bodies",
        "image_path": optical_geotiff,
        "box_threshold": 0.3,
        "text_threshold": 0.25
    })
    assert hasattr(grd_out, "geojson")
    assert grd_out.mode == "stub"

    # 4. change_vqa
    cvqa_out = registry.execute("change_vqa", {
        "query": "What changed?",
        "image_t1_path": optical_geotiff,
        "image_t2_path": optical_geotiff
    })
    assert hasattr(cvqa_out, "answer")
    assert cvqa_out.mode == "stub"

    # 5. change_map
    cmap_out = registry.execute("change_map", {
        "image_t1_path": optical_geotiff,
        "image_t2_path": optical_geotiff,
        "threshold": 0.5,
        "min_area_m2": 100.0
    })
    assert hasattr(cmap_out, "change_mask_path")
    assert cmap_out.mode == "stub"

    # 6. optical_sar_fusion
    fusion_out = registry.execute("optical_sar_fusion", {
        "optical_path": optical_geotiff,
        "sar_path": sar_geotiff,
        "target_classes": ["water", "built_up"],
        "confidence_threshold": 0.5
    })
    assert hasattr(fusion_out, "classification_map_path")
    assert fusion_out.mode == "stub"

    # 7. spectral_index
    spec_out = registry.execute("spectral_index", {
        "image_path": optical_geotiff,
        "index_type": "NDVI",
        "threshold": 0.2
    })
    assert hasattr(spec_out, "mean_index")
    assert spec_out.mode == "stub"
