from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .api.routes import router as api_router
from .api.batch import router as batch_router

settings = get_settings()

app = FastAPI(
    title=settings.app.name,
    version=settings.version,
    description="Agentic Vision-Language Assistant for Remote Sensing Image Analysis (ISRO/SAC 26167)"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs and uploads static folders
outputs_dir = Path(settings.app.outputs_dir)
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

storage_dir = Path(settings.app.storage_dir)
storage_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/uploads", StaticFiles(directory=str(storage_dir)), name="uploads")

# Include API routers
app.include_router(api_router, prefix="/api")
app.include_router(batch_router, prefix="/api")

@app.get("/")
def root():
    return {
        "message": "SatQuery AI Backend API is running.",
        "docs": "/docs",
        "health": "/api/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.app.host, port=settings.app.port, reload=True)
