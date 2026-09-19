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
    tiles_used: List[Dict[str, Any]] = Field(default_factory=list)
    rendering_applied: Optional[str] = None
    mode: str = "model"

# RSCaption
class RSCaptionInput(BaseModel):
    image_path: str
    max_length: int = 100
    style: str = "detailed"

class RSCaptionOutput(BaseModel):
    caption: str
    tags: List[str] = Field(default_factory=list)
    confidence: float
    rendering_applied: Optional[str] = None
    mode: str = "model"

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
    tiles_used: List[Dict[str, Any]] = Field(default_factory=list)
    rendering_applied: Optional[str] = None
    mode: str = "model"

# ChangeVQA
class ChangeVQAInput(BaseModel):
    query: str
    image_t1_path: str
    image_t2_path: str
    change_mask_path: Optional[str] = None
    change_stats: Optional[Dict[str, Any]] = None
    design_mode: Optional[str] = None

class ChangeVQAOutput(BaseModel):
    answer: str
    confidence: float
    change_summary: str
    direction: Optional[str] = None
    mask_verified: bool = True
    vlm_cross_check_agreed: Optional[bool] = None
    mode: str = "model"

# ChangeMap
class ChangeMapInput(BaseModel):
    image_t1_path: str
    image_t2_path: str
    threshold: float = 0.45
    min_area_m2: float = 100.0

class ChangeMapOutput(BaseModel):
    change_mask_path: str
    area_changed_m2: float = 0.0
    area_changed_ha: float = 0.0
    area_changed_km2: float = 0.0
    percentage_changed: float = 0.0
    direction_of_change: str
    per_class_change: Optional[Dict[str, float]] = None
    geojson: Optional[GeoJSONFeatureCollection] = None
    mode: str = "model"

# OpticalSARFusion
class OpticalSARFusionInput(BaseModel):
    optical_path: str
    sar_path: str
    target_classes: List[str] = Field(default_factory=lambda: ["water", "built_up"])
    confidence_threshold: float = 0.5
    ablation_mode: str = "fused"  # "fused", "optical_only", "sar_only"
    sensor_profile: Optional[str] = None

class OpticalSARFusionOutput(BaseModel):
    classification_map_path: str
    water_area_m2: float = 0.0
    water_area_ha: float = 0.0
    water_area_km2: float = 0.0
    water_percentage: float = 0.0
    built_up_area_m2: float = 0.0
    built_up_area_ha: float = 0.0
    built_up_area_km2: float = 0.0
    built_up_percentage: float = 0.0
    geojson: Optional[GeoJSONFeatureCollection] = None
    ablation_mode: str = "fused"
    degradation_mode: Optional[str] = None
    resampling_info: Optional[Dict[str, Any]] = None
    mode: str = "model"

# SpectralIndex
class SpectralIndexInput(BaseModel):
    image_path: str
    index_type: str = "NDWI"  # NDVI, NDWI, NDBI
    threshold: float = 0.0
    sensor_profile: Optional[str] = "sentinel2"

class SpectralIndexOutput(BaseModel):
    index_type: str
    status: str = "computed"  # "computed" or "not_applicable"
    reason: Optional[str] = None
    index_map_path: Optional[str] = None
    mean_index: Optional[float] = None
    positive_area_m2: float = 0.0
    positive_area_ha: float = 0.0
    positive_area_km2: float = 0.0
    positive_percentage: float = 0.0
    geojson: Optional[GeoJSONFeatureCollection] = None
    mode: str = "model"
