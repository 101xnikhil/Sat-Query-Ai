from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class Modality(str, Enum):
    OPTICAL = "optical"
    MULTISPECTRAL = "multispectral"
    SAR = "sar"
    UNKNOWN = "unknown"

class TaskType(str, Enum):
    SINGLE_VQA = "single_vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE_VQA = "change_vqa"
    CHANGE_MAP = "change_map"
    OPTICAL_SAR_ANALYSIS = "optical_sar_analysis"

class RasterFormat(str, Enum):
    GEOTIFF = "geotiff"
    TIFF = "tiff"
    PNG = "png"
    JPEG = "jpeg"

class GeoBBox(BaseModel):
    minx: float
    miny: float
    maxx: float
    maxy: float
    crs: str = "EPSG:4326"

class ImageMetadata(BaseModel):
    image_id: str
    filename: str
    file_path: str
    format: RasterFormat
    modality: Modality = Modality.OPTICAL
    width: int
    height: int
    crs: Optional[str] = None
    bounds: Optional[GeoBBox] = None
    resolution: Optional[List[float]] = None
    band_count: int = 3
    dtype: str = "uint8"
    nodata: Optional[float] = None
    acquisition_date: Optional[str] = None
    is_georeferenced: bool = True
    transform: Optional[List[float]] = None
    modality_confidence: Optional[float] = 1.0
    sar_metadata: Optional[Dict[str, Any]] = None

class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: Any

class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: GeoJSONGeometry
    properties: Dict[str, Any] = Field(default_factory=dict)

class GeoJSONFeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: List[GeoJSONFeature] = Field(default_factory=list)
