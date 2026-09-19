const API_BASE = '/api';

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function uploadImagery(files, modality = null, acquisitionDate = null) {
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }
  if (modality) formData.append('modality', modality);
  if (acquisitionDate) formData.append('acquisition_date', acquisitionDate);

  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Upload failed');
  }
  return res.json();
}

export async function sendQuery(query, imageIds, taskOverride = null, sessionId = null) {
  const payload = {
    query,
    image_ids: imageIds,
    task_override: taskOverride,
    session_id: sessionId,
  };

  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Query failed');
  }
  return res.json();
}

export async function fetchTrace(sessionId) {
  const res = await fetch(`${API_BASE}/trace/${sessionId}`);
  if (!res.ok) throw new Error(`Failed to fetch trace: ${res.statusText}`);
  return res.json();
}
