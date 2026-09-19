from typing import List, Type, Dict, Any, Optional
from pydantic import BaseModel
from .base import BaseTool
from ..schemas.tools import ChangeMapInput, ChangeMapOutput
from models.inference.change_detector import compute_radiometric_change

class ChangeMapTool(BaseTool):
    """
    Real Change Map Tool.
    Performs Radiometric Change Vector Analysis (RCVA) between co-registered multi-temporal rasters,
    producing a georeferenced GeoTIFF change mask, vector GeoJSON polygons, and empirical statistics.
    """
    name: str = "change_map"
    description: str = "Generates pixel-level change mask and calculates empirical area changed and direction."
    input_schema: Type[BaseModel] = ChangeMapInput
    output_schema: Type[BaseModel] = ChangeMapOutput
    whitelisted_params: List[str] = ["image_t1_path", "image_t2_path", "threshold", "min_area_m2"]

    def run(self, input_data: ChangeMapInput) -> ChangeMapOutput:
        res = compute_radiometric_change(
            image_t1_path=input_data.image_t1_path,
            image_t2_path=input_data.image_t2_path,
            threshold=input_data.threshold,
            min_area_m2=input_data.min_area_m2
        )
        return ChangeMapOutput(
            change_mask_path=res["change_mask_path"],
            area_changed_m2=res.get("area_changed_m2", 0.0),
            area_changed_ha=res.get("area_changed_ha", 0.0),
            area_changed_km2=res["area_changed_km2"],
            percentage_changed=res["percentage_changed"],
            direction_of_change=res["direction_of_change"],
            per_class_change=res.get("per_class_change"),
            geojson=res["geojson"],
            mode="model"
        )
