from typing import List, Type, Dict, Any, Optional
from pathlib import Path
import numpy as np
import rasterio
from pydantic import BaseModel

from ..base import BaseTool
from ...schemas.tools import SpectralIndexInput, SpectralIndexOutput
from ...schemas.common import GeoJSONFeatureCollection, GeoJSONFeature, GeoJSONGeometry
from ...ingestion.reader import read_image_metadata, read_raster_array

class SpectralIndexTool(BaseTool):
    """
    Spectral Index Tool with strict band-presence checking.
    Calculates NDWI, NDVI, NDBI strictly based on required bands.
    If required bands are missing, returns status='not_applicable' cleanly.
    """
    name: str = "spectral_index"
    description: str = "Computes spectral indices (NDVI, NDWI, NDBI) with strict band checking."
    input_schema: Type[BaseModel] = SpectralIndexInput
    output_schema: Type[BaseModel] = SpectralIndexOutput
    whitelisted_params: List[str] = ["image_path", "index_type", "threshold", "sensor_profile"]

    def run(self, input_data: SpectralIndexInput) -> SpectralIndexOutput:
        from ..spectral_index import SpectralIndexTool as RealSpectralIndexTool
        real_tool = RealSpectralIndexTool()
        return real_tool.run(input_data)
