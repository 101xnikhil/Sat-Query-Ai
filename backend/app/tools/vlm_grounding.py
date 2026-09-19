from typing import List, Type
from pydantic import BaseModel
from .base import BaseTool
from ..schemas.tools import RSGroundingInput, RSGroundingOutput
from models.inference.vlm_wrapper import get_vlm_wrapper
from models.inference.georeferencer import project_boxes_to_geojson

class RSGroundingTool(BaseTool):
    """
    Real Text-Guided Region Grounding Tool.
    Detects target objects/features from natural language queries,
    and maps them into georeferenced WGS84 (EPSG:4326) GeoJSON polygons using rasterio.
    """
    name: str = "rs_grounding"
    description: str = "Locates and grounds specified features into georeferenced GeoJSON polygons."
    input_schema: Type[BaseModel] = RSGroundingInput
    output_schema: Type[BaseModel] = RSGroundingOutput
    whitelisted_params: List[str] = ["query", "image_path", "box_threshold", "text_threshold"]

    def __init__(self):
        self.vlm = get_vlm_wrapper()

    def run(self, input_data: RSGroundingInput) -> RSGroundingOutput:
        # 1. Detect candidate bounding boxes using VLM inference wrapper with tiling & NMS
        boxes, tiles_used, render_mode = self.vlm.ground_query(
            query=input_data.query,
            image_path=input_data.image_path,
            box_threshold=input_data.box_threshold,
            return_metadata=True
        )

        # 2. Project bounding boxes into georeferenced GeoJSON features
        geojson = project_boxes_to_geojson(
            boxes=boxes,
            raster_path=input_data.image_path,
            query=input_data.query
        )

        labels = [b.label for b in boxes]
        confidences = [b.confidence for b in boxes]

        return RSGroundingOutput(
            geojson=geojson,
            detected_count=len(boxes),
            labels=labels,
            confidences=confidences,
            tiles_used=tiles_used,
            rendering_applied=render_mode
        )
