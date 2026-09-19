from typing import List, Type, Dict, Any, Optional
from pydantic import BaseModel

from .base import BaseTool
from ..schemas.tools import OpticalSARFusionInput, OpticalSARFusionOutput
from models.inference.fusion_segmenter import OpticalSARFusionEngine

class OpticalSARFusionTool(BaseTool):
    """
    Real Optical + SAR Two-Stream Fusion Tool.
    Performs cross-modal segmentation with overlap tiling, cosine blending,
    empirical surface area calculations, georeferenced GeoTIFF creation,
    and GeoJSON polygon extraction.
    """
    name: str = "optical_sar_fusion"
    description: str = "Two-stream optical and SAR cross-modal segmentation for water bodies and built-up areas with overlap tiling."
    input_schema: Type[BaseModel] = OpticalSARFusionInput
    output_schema: Type[BaseModel] = OpticalSARFusionOutput
    whitelisted_params: List[str] = ["optical_path", "sar_path", "target_classes", "confidence_threshold"]

    def __init__(self, tile_size: int = 128, overlap: int = 32):
        super().__init__()
        self.engine = OpticalSARFusionEngine(tile_size=tile_size, overlap=overlap)

    def run(self, input_data: OpticalSARFusionInput) -> OpticalSARFusionOutput:
        results = self.engine.segment_scene(
            optical_path=input_data.optical_path,
            sar_path=input_data.sar_path,
            target_classes=input_data.target_classes,
            confidence_threshold=input_data.confidence_threshold,
            output_dir="data/outputs"
        )

        return OpticalSARFusionOutput(
            classification_map_path=results["classification_map_path"],
            water_area_km2=results["water_area_km2"],
            built_up_area_km2=results["built_up_area_km2"],
            water_percentage=results["water_percentage"],
            built_up_percentage=results["built_up_percentage"],
            geojson=results["geojson"]
        )
