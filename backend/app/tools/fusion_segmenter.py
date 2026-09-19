from typing import List, Type, Dict, Any, Optional
from pydantic import BaseModel

from .base import BaseTool
from ..schemas.tools import OpticalSARFusionInput, OpticalSARFusionOutput
from models.inference.two_stream_fusion import TwoStreamFusionEngine

class OpticalSARFusionTool(BaseTool):
    """
    Real Optical + SAR Two-Stream Fusion Tool.
    Performs cross-modal segmentation with separate optical and SAR encoders,
    feature-level fusion, overlap tiling, cosine blending,
    empirical surface area calculations (m², ha, km², %), georeferenced GeoTIFF creation,
    GeoJSON polygon extraction, and ablation support (optical_only, sar_only, fused).
    """
    name: str = "optical_sar_fusion"
    description: str = "Two-stream optical and SAR cross-modal segmentation for water bodies and built-up areas with overlap tiling and ablation modes."
    input_schema: Type[BaseModel] = OpticalSARFusionInput
    output_schema: Type[BaseModel] = OpticalSARFusionOutput
    whitelisted_params: List[str] = [
        "optical_path", "sar_path", "target_classes",
        "confidence_threshold", "ablation_mode", "sensor_profile"
    ]

    def __init__(self, config_path: str = "configs/optical_sar_fusion.yaml", tile_size: int = 128, overlap: int = 32):
        super().__init__()
        self.engine = TwoStreamFusionEngine(config_path=config_path, tile_size=tile_size, overlap=overlap)

    def run(self, input_data: OpticalSARFusionInput) -> OpticalSARFusionOutput:
        ablation = input_data.ablation_mode or "fused"
        results = self.engine.segment_scene(
            optical_path=input_data.optical_path,
            sar_path=input_data.sar_path,
            target_classes=input_data.target_classes,
            confidence_threshold=input_data.confidence_threshold,
            ablation_mode=ablation,
            output_dir="data/outputs"
        )

        return OpticalSARFusionOutput(
            classification_map_path=results["classification_map_path"],
            water_area_m2=results.get("water_area_m2", 0.0),
            water_area_ha=results.get("water_area_ha", 0.0),
            water_area_km2=results.get("water_area_km2", 0.0),
            water_percentage=results.get("water_percentage", 0.0),
            built_up_area_m2=results.get("built_up_area_m2", 0.0),
            built_up_area_ha=results.get("built_up_area_ha", 0.0),
            built_up_area_km2=results.get("built_up_area_km2", 0.0),
            built_up_percentage=results.get("built_up_percentage", 0.0),
            geojson=results.get("geojson"),
            ablation_mode=ablation,
            degradation_mode=results.get("degradation_mode", "nominal"),
            resampling_info=results.get("resampling_info"),
            mode="model"
        )
