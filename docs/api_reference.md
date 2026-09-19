# SatQuery AI REST API Reference
**ISRO/SAC Problem Statement 26167**

SatQuery AI provides a robust, observable FastAPI interface for satellite imagery ingestion, agentic orchestration, and evidence-grounded reporting.

Base URL: `http://localhost:8000/api`

---

## 1. System Health & Tool Discovery

### `GET /api/health`
Inspects operational readiness, storage write permissions, caching layer, and lists registered tools.

**Response `200 OK`**:
```json
{
  "status": "healthy",
  "app": "SatQuery AI",
  "version": "1.0.0",
  "registered_tools": [
    "rs_vqa",
    "rs_caption",
    "rs_grounding",
    "change_vqa",
    "change_map",
    "optical_sar_fusion",
    "spectral_index"
  ],
  "model_status": "ready",
  "cache_status": "active",
  "storage_writable": true
}
```

---

## 2. Satellite Imagery Ingestion

### `POST /api/upload`
Uploads 1 or 2 remote sensing imagery files (`.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`).
Extracts geospatial metadata, CRS, resolution, acquisition dates, and detects sensor modality (Optical vs SAR).

**Request (Multipart Form)**:
- `files`: One or more binary image files (Max 200MB per file).
- `modality` *(optional)*: `"optical"` | `"multispectral"` | `"sar"` manual override.
- `acquisition_date` *(optional)*: ISO 8601 acquisition timestamp (e.g. `2023-05-01T10:30:00Z`).

**Response `200 OK`**:
```json
{
  "images": [
    {
      "image_id": "sentinel2_scene_a1b2c3",
      "filename": "sentinel2_scene.tif",
      "file_path": "data/uploads/sentinel2_scene_a1b2c3.tif",
      "format": "geotiff",
      "width": 512,
      "height": 512,
      "crs": "EPSG:4326",
      "band_count": 4,
      "dtype": "uint16",
      "resolution": [10.0, 10.0],
      "modality": "optical",
      "modality_confidence": 0.98,
      "acquisition_date": "2023-05-01T10:30:00Z"
    }
  ],
  "message": "Successfully uploaded and processed 1 imagery file(s)."
}
```

**Errors**:
- `400 Bad Request`: Unpermitted file extension.
- `413 Payload Too Large`: Upload file exceeds 200MB.

---

## 3. Natural Language Query & Orchestration

### `POST /api/query`
Executes agentic multi-step orchestration across provided imagery IDs.

**Request JSON**:
```json
{
  "session_id": "optional_session_uuid",
  "query": "what changed and is the new area water or built-up?",
  "image_ids": [
    "sentinel2_scene_a1b2c3",
    "sentinel1_sar_d4e5f6"
  ],
  "task_override": null
}
```

**Response `200 OK`**:
```json
{
  "session_id": "session_8f9e1d2c3b",
  "query": "what changed and is the new area water or built-up?",
  "task": "change_vqa",
  "answer": "Multitemporal change analysis confirms 12,450.0 m² (1.245 ha, 4.8% of scene) underwent change.\n• Multi-step Land-Cover Breakdown: Of the changed area, 8,200.0 m² (65.9%) is surface water and 4,250.0 m² (34.1%) is built-up urban structures.",
  "layers": [
    {
      "layer_id": "change_9c8b7a",
      "name": "Temporal Change Footprint",
      "type": "geojson",
      "color": "#f43f5e",
      "opacity": 0.65
    }
  ],
  "computed_metrics": {
    "area_changed_m2": 12450.0,
    "area_changed_ha": 1.245,
    "percentage_changed": 4.8,
    "direction_of_change": "vegetation decrease / darkening",
    "changed_water_area_m2": 8200.0,
    "changed_water_percentage": 65.9,
    "changed_built_up_area_m2": 4250.0,
    "changed_built_up_percentage": 34.1
  },
  "confidence_score": 0.9425,
  "trace": {
    "session_id": "session_8f9e1d2c3b",
    "task": "change_vqa",
    "fallback_used": false,
    "stage_latencies": {
      "ingestion_ms": 1.2,
      "validation_ms": 3.4,
      "planning_ms": 0.2,
      "tool_execution_ms": 112.5,
      "synthesis_ms": 0.8
    },
    "confidence_breakdown": {
      "overall": 0.9425,
      "token_probability": 0.90,
      "cross_tool_agreement": 1.0,
      "input_quality": 1.0
    },
    "tool_calls": [
      {
        "tool_name": "change_map",
        "parameters": {"threshold": 0.45, "min_area_m2": 100.0},
        "status": "success",
        "duration_ms": 32.1
      },
      {
        "tool_name": "optical_sar_fusion",
        "parameters": {"target_classes": ["water", "built_up"], "confidence_threshold": 0.5},
        "status": "success",
        "duration_ms": 65.3
      },
      {
        "tool_name": "change_vqa",
        "parameters": {"query": "what changed and is the new area water or built-up?"},
        "status": "success",
        "duration_ms": 15.1
      }
    ],
    "total_duration_ms": 125.4
  },
  "pdf_report_url": "/api/reports/session_8f9e1d2c3b/pdf"
}
```

---

## 4. Execution Audit Trace

### `GET /api/trace/{run_id}`
Retrieves the immutable audit trace for a specific execution run.

---

## 5. Intelligence Report Export

### `GET /api/reports/{session_id}/pdf`
Downloads the publication-ready PDF intelligence report generated for the session.

**Response**: Binary PDF file (`Content-Type: application/pdf`).
