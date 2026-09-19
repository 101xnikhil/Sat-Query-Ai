from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from .common import GeoJSONFeatureCollection

# RSVQA
class RSVQAInput(BaseModel):
    query: str
    image_path: str
    confidence_threshold: float = 0.5

class RSVQAOutput(BaseModel):
    answer: str
    confidence: float
    evidence_summary: str

# RSCaption
class RSCaptionInput(BaseModel):
    image_path: str
    max_length: int = 100
    style: str = "detailed"

class RSCaptionOutput(BaseModel):
    caption: str
    tags: List[str] = Field(default_factory=list)
    confidence: float

# RSGrounding
class RSGroundingInput(BaseModel):
    query: str
    image_path: str
    box_threshold: float = 0.3
    text_threshold: float = 0.25

class RSGroundingOutput(BaseModel):
    geojson: GeoJSONFeatureCollection
    detected_count: int
    labels: List[str] = Field(default_factory=list)
    confidences: List[float] = Field(default_factory=list)

# ChangeVQA
class ChangeVQAInput(BaseModel):
    query: str
    image_t1_path: str
    image_t2_path: str

class ChangeVQAOutput(BaseModel):
    answer: str
    confidence: float
    change_summary: str

# ChangeMap
class ChangeMapInput(BaseModel):
    image_t1_path: str
    image_t2_path: str
    threshold: float = 0.5
    min_area_m2: float = 100.0

class ChangeMapOutput(BaseModel):
    change_mask_path: str
    area_changed_km2: float
    percentage_changed: float
    direction_of_change: str
    geojson: Optional[GeoJSONFeatureCollection] = None

# OpticalSARFusion
class OpticalSARFusionInput(BaseModel):
    optical_path: str
    sar_path: str
    target_classes: List[str] = Field(default_factory=lambda: ["water", "built_up"])
    confidence_threshold: float = 0.5

class OpticalSARFusionOutput(BaseModel):
    classification_map_path: str
    water_area_km2: float
    built_up_area_km2: float
    water_percentage: float
    built_up_percentage: float
    geojson: Optional[GeoJSONFeatureCollection] = None

# SpectralIndex
class SpectralIndexInput(BaseModel):
    image_path: str
    index_type: str = "NDVI"  # NDVI, NDWI, NDBI
    threshold: float = 0.2

class SpectralIndexOutput(BaseModel):
    index_type: str
    index_map_path: str
    mean_index: float
    positive_area_km2: float
    positive_percentage: float
    geojson: Optional[GeoJSONFeatureCollection] = None
