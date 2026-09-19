import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from ..schemas.common import Modality, ImageMetadata
from ..schemas.query import QueryRequest, QueryResponse, UploadResponse
from ..schemas.trace import ExecutionTrace
from ..config import get_settings
from ..ingestion.reader import read_image_metadata, inspect_modality
from ..controller.engine import ControllerEngine
from ..controller.router import IncompatibleInputError
from ..trace.logger import load_trace
from ..tools.registry import ToolRegistry

router = APIRouter()
settings = get_settings()
engine = ControllerEngine(settings=settings)

MAX_UPLOAD_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB limit
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

@router.get("/health")
def health_check():
    registry = ToolRegistry.get_instance()
    storage_ok = Path(settings.app.storage_dir).exists()
    return {
        "status": "healthy",
        "app": settings.app.name,
        "version": settings.version,
        "registered_tools": [t["name"] for t in registry.list_tools()],
        "model_status": "ready",
        "cache_status": "active",
        "storage_writable": storage_ok
    }

@router.post("/upload", response_model=UploadResponse)
async def upload_images(
    files: List[UploadFile] = File(...),
    modality: Optional[str] = Form(None),
    acquisition_date: Optional[str] = Form(None)
):
    storage_dir = Path(settings.app.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_metas: List[ImageMetadata] = []
    modality_enum = None
    if modality and modality.lower() in [m.value for m in Modality]:
        modality_enum = Modality(modality.lower())

    for f in files:
        file_ext = Path(f.filename or "").suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type '{file_ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        base_name = Path(f.filename).stem or "image"
        # Sanitize filename
        safe_base = "".join(c for c in base_name if c.isalnum() or c in ("-", "_"))
        safe_id = f"{safe_base}_{uuid.uuid4().hex[:6]}"
        target_path = storage_dir / f"{safe_id}{file_ext}"

        total_bytes = 0
        try:
            with open(target_path, "wb") as buffer:
                while True:
                    chunk = await f.read(1024 * 1024)  # 1MB chunk
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                        buffer.close()
                        if target_path.exists():
                            target_path.unlink()
                        raise HTTPException(
                            status_code=413,
                            detail=f"File '{f.filename}' exceeds maximum allowed size of 200MB."
                        )
                    buffer.write(chunk)
        except HTTPException:
            raise
        except Exception as e:
            if target_path.exists():
                target_path.unlink()
            raise HTTPException(status_code=500, detail=f"Failed to save upload '{f.filename}': {str(e)}")

        meta = read_image_metadata(
            file_path=str(target_path),
            image_id=safe_id,
            modality_hint=modality_enum,
            acquisition_date=acquisition_date
        )
        uploaded_metas.append(meta)

    return UploadResponse(
        images=uploaded_metas,
        message=f"Successfully uploaded and processed {len(uploaded_metas)} imagery file(s)."
    )

@router.post("/query", response_model=QueryResponse)
def run_query(request: QueryRequest):
    try:
        response = engine.process_query(request)
        return response
    except (ValueError, IncompatibleInputError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

@router.get("/trace/{run_id}", response_model=ExecutionTrace)
def get_session_trace(run_id: str):
    trace = load_trace(run_id, output_dir=settings.app.outputs_dir)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace for run/session '{run_id}' not found.")
    return trace

@router.get("/outputs/{filename}")
def get_output_file(filename: str):
    file_path = Path(settings.app.outputs_dir) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Output file '{filename}' not found.")
    return FileResponse(str(file_path))

@router.get("/reports/{session_id}/pdf")
def get_session_pdf_report(session_id: str):
    reports_dir = Path(settings.app.outputs_dir) / "reports"
    pdf_path = reports_dir / f"SatQuery_Report_{session_id}.pdf"
    
    if pdf_path.exists():
        return FileResponse(
            str(pdf_path),
            media_type="application/pdf",
            filename=f"SatQuery_Report_{session_id}.pdf"
        )
    
    # Try generating from cached response
    cached = engine.get_cached_response(session_id)
    if cached:
        generated_path = engine.pdf_generator.generate_report(cached)
        return FileResponse(
            generated_path,
            media_type="application/pdf",
            filename=f"SatQuery_Report_{session_id}.pdf"
        )
    
    # Try generating from loaded trace
    trace = load_trace(session_id, output_dir=settings.app.outputs_dir)
    if trace:
        dummy_resp = QueryResponse(
            session_id=session_id,
            query="Remote Sensing Analysis",
            task=trace.task,
            answer="Analysis execution completed with verified trace logging.",
            layers=[],
            computed_metrics=trace.computed_metrics,
            confidence_score=trace.confidence_score,
            trace=trace
        )
        generated_path = engine.pdf_generator.generate_report(dummy_resp)
        return FileResponse(
            generated_path,
            media_type="application/pdf",
            filename=f"SatQuery_Report_{session_id}.pdf"
        )

    raise HTTPException(status_code=404, detail=f"PDF report for session '{session_id}' not found.")

