import pytest
import io
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health_endpoint_enriched():
    """Verify health endpoint reports subsystems and registered tools."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "registered_tools" in data
    assert len(data["registered_tools"]) >= 7
    assert data["model_status"] == "ready"
    assert data["cache_status"] == "active"
    assert data["storage_writable"] is True

def test_upload_rejects_unallowed_file_extension():
    """Verify upload rejects invalid file extensions with 400 Bad Request."""
    fake_file = io.BytesIO(b"fake dangerous executable content")
    resp = client.post(
        "/api/upload",
        files=[("files", ("malicious.exe", fake_file, "application/x-msdownload"))]
    )
    assert resp.status_code == 400
    assert "Invalid file type" in resp.json()["detail"]

def test_upload_rejects_unallowed_script_extension():
    """Verify upload rejects .sh or .py scripts."""
    fake_file = io.BytesIO(b"#!/bin/bash\nrm -rf /")
    resp = client.post(
        "/api/upload",
        files=[("files", ("exploit.sh", fake_file, "text/x-sh"))]
    )
    assert resp.status_code == 400
    assert "Invalid file type" in resp.json()["detail"]
