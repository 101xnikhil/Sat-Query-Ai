import React, { useState, useEffect } from 'react';
import { Satellite, Activity, ShieldCheck, Download } from 'lucide-react';
import UploadPanel from './components/UploadPanel';
import MapViewer from './components/MapViewer';
import ChatPanel from './components/ChatPanel';
import { checkHealth, sendQuery } from './services/api';

export default function App() {
  const [currentMode, setCurrentMode] = useState('single');
  const [uploadedImages, setUploadedImages] = useState([]);
  const [layers, setLayers] = useState([]);
  const [queryResponse, setQueryResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState('connecting');

  // Verify backend health on startup
  useEffect(() => {
    checkHealth()
      .then(() => setBackendStatus('connected'))
      .catch(() => setBackendStatus('disconnected'));
  }, []);

  const handleModeChange = (mode) => {
    setCurrentMode(mode);
    setUploadedImages([]);
    setLayers([]);
    setQueryResponse(null);
    setError(null);
  };

  const handleUploadComplete = (images) => {
    setUploadedImages((prev) => [...prev, ...images]);
    setError(null);
  };

  const handleRunQuery = async (queryText) => {
    setLoading(true);
    setError(null);

    try {
      const imageIds = uploadedImages.map((img) => img.image_id);
      const res = await sendQuery(queryText, imageIds);
      setQueryResponse(res);

      // Merge newly returned layers
      if (res.layers && res.layers.length > 0) {
        setLayers(res.layers);
      }
    } catch (err) {
      setError(err.message || 'Query execution failed');
    } finally {
      setLoading(false);
    }
  };

  const handleLayerToggle = (layerId) => {
    setLayers((prev) =>
      prev.map((l) => (l.layer_id === layerId ? { ...l, visible: !l.visible } : l))
    );
  };

  return (
    <div className="app-container">
      {/* Top Aerospace Navigation Bar */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-icon">
            <Satellite size={22} color="#070b14" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="brand-title">SatQuery AI</span>
              <span className="brand-badge">ISRO / SAC 26167</span>
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              Agentic Vision-Language Assistant for Remote Sensing Analysis
            </div>
          </div>
        </div>

        <div className="header-status">
          <div className="status-indicator">
            <span
              className="status-dot"
              style={{
                backgroundColor: backendStatus === 'connected' ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                boxShadow: backendStatus === 'connected' ? '0 0 8px var(--accent-emerald)' : '0 0 8px var(--accent-rose)'
              }}
            />
            <span>Backend: {backendStatus === 'connected' ? 'Online' : 'Reconnecting...'}</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Phase 1 Scaffold
          </div>
        </div>
      </header>

      {/* 3-Column Main Workspace */}
      <main className="workspace-grid">
        <UploadPanel
          currentMode={currentMode}
          onModeChange={handleModeChange}
          uploadedImages={uploadedImages}
          onUploadComplete={handleUploadComplete}
        />

        <MapViewer
          uploadedImages={uploadedImages}
          layers={layers}
          onLayerToggle={handleLayerToggle}
        />

        <ChatPanel
          currentMode={currentMode}
          uploadedImages={uploadedImages}
          onRunQuery={handleRunQuery}
          queryResponse={queryResponse}
          loading={loading}
          error={error}
        />
      </main>
    </div>
  );
}
