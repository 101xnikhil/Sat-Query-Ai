import pytest
import numpy as np
from backend.rendering.compositor import (
    percentile_stretch,
    render_multispectral_rgb,
    render_false_color_cir,
    render_sar_db,
    auto_render_for_vlm
)

def test_percentile_stretch_scaling():
    arr = np.array([0.0, 10.0, 50.0, 100.0, 1000.0])
    stretched = percentile_stretch(arr, p_min=2.0, p_max=98.0)
    assert stretched.dtype == np.uint8
    assert stretched.min() >= 0
    assert stretched.max() <= 255
    assert stretched[-1] == 255

def test_multispectral_true_color_and_cir_rendering(optical_geotiff):
    rgb, mode_rgb = auto_render_for_vlm(optical_geotiff, query="Identify urban buildings")
    assert rgb.ndim == 3
    assert rgb.shape[2] == 3
    assert rgb.dtype == np.uint8
    assert mode_rgb == "multispectral_true_color_rgb"

    cir, mode_cir = auto_render_for_vlm(optical_geotiff, query="Analyze vegetation and forest canopy")
    assert cir.ndim == 3
    assert cir.shape[2] == 3
    assert cir.dtype == np.uint8
    assert mode_cir in ["false_color_cir", "multispectral_false_color_cir"]

def test_sar_linear_to_db_rendering(sar_geotiff):
    sar_img, mode_sar = auto_render_for_vlm(sar_geotiff, query="Detect water bodies")
    assert sar_img.ndim == 3
    assert sar_img.shape[2] == 3
    assert sar_img.dtype == np.uint8
    assert "sar_db" in mode_sar
