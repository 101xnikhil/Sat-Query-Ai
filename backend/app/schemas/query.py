from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from .common import TaskType, ImageMetadata, GeoJSONFeatureCollection
from .trace import ExecutionTrace

class UploadResponse(BaseModel):
    images: List[ImageMetadata]
    message: str

class LayerItem(BaseModel):
    layer_id: str
    name: str
    type: str  # "geojson", "raster_mask", "raster_tile"
    geojson: Optional[GeoJSONFeatureCollection] = None
    mask_url: Optional[str] = None
    color: str = "#00FFCC"
    visible: bool = True
    opacity: float = 0.7
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryRequest(BaseModel):
    query: str
    image_ids: List[str]
    task_override: Optional[TaskType] = None
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    session_id: str
    query: str
    task: TaskType
    answer: str
    layers: List[LayerItem] = Field(default_factory=list)
    computed_metrics: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float
    trace: ExecutionTrace
    pdf_report_url: Optional[str] = None
