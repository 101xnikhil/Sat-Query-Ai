import React, { useState, useEffect } from 'react';
import { 
  Satellite, 
  Activity, 
  ShieldCheck, 
  Layers, 
  Cpu, 
  Sparkles, 
  Database,
  Compass,
  Zap,
  Target,
  Clock,
  Radio
} from 'lucide-react';
import UploadPanel from './components/UploadPanel';
import MapViewer from './components/MapViewer';
import ChatPanel from './components/ChatPanel';
import { checkHealth, sendQuery, getPresets, loadPreset } from './services/api';

export default function App() {
  const [currentMode, setCurrentMode] = useState('single');
  const [uploadedImages, setUploadedImages] = useState([]);
  const [layers, setLayers] = useState([]);
  const [queryResponse, setQueryResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState('connecting');
  const [activePreset, setActivePreset] = useState(null);
  const [presetLoading, setPresetLoading] = useState(false);
  const [defaultPrompt, setDefaultPrompt] = useState('');

  // Check health and telemetry on startup
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
    setActivePreset(null);
    setDefaultPrompt('');
  };

  const handleUploadComplete = (images) => {
    setUploadedImages((prev) => [...prev, ...images]);
    setError(null);
  };

  const handleClearImages = () => {
    setUploadedImages([]);
    setLayers([]);
    setQueryResponse(null);
    setError(null);
    setActivePreset(null);
    setDefaultPrompt('');
  };

  const handleLoadPreset = async (presetId) => {
    setPresetLoading(true);
    setError(null);
    try {
      const data = await loadPreset(presetId);
      setActivePreset(presetId);
      setCurrentMode(data.mode);
      setUploadedImages(data.images);
      setDefaultPrompt(data.default_query);
      setLayers([]);
      setQueryResponse(null);
    } catch (err) {
      setError(err.message || `Failed to load preset ${presetId}`);
    } finally {
      setPresetLoading(false);
    }
  };

  const handleRunQuery = async (queryText) => {
    setLoading(true);
    setError(null);

    try {
      const imageIds = uploadedImages.map((img) => img.image_id);
      const res = await sendQuery(queryText, imageIds);
      setQueryResponse(res);

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
      {/* Aerospace Mission Command Header */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-icon-wrapper">
            <div className="brand-icon-orbit" />
            <div className="brand-icon">
              <Satellite size={20} />
            </div>
          </div>
          <div>
            <div className="brand-title">
              SatQuery AI
            </div>
            <div className="brand-subtitle">
              Tactical Vision-Language Assistant for Remote Sensing Image Analysis
            </div>
          </div>
        </div>

        {/* 1-Click Capabilities Preset Launcher */}
        <div className="header-presets">
          <span className="presets-label">⚡ Capabilities:</span>
          <button
            className={`preset-chip-btn ${activePreset === 'single_vqa' ? 'active' : ''}`}
            onClick={() => handleLoadPreset('single_vqa')}
            title="Single-Image VQA & Land-Cover Understanding"
          >
            <Compass size={13} color="var(--cyan-primary)" />
            <span>Single VQA</span>
          </button>
          <button
            className={`preset-chip-btn ${activePreset === 'grounding' ? 'active' : ''}`}
            onClick={() => handleLoadPreset('grounding')}
            title="Text-Guided Spatial Grounding to EPSG:4326 GeoJSON"
          >
            <Target size={13} color="var(--emerald-primary)" />
            <span>Grounding</span>
          </button>
          <button
            className={`preset-chip-btn ${activePreset === 'change' ? 'active' : ''}`}
            onClick={() => handleLoadPreset('change')}
            title="Bi-Temporal Change Detection & Directional Dynamics"
          >
            <Clock size={13} color="var(--rose-primary)" />
            <span>Bi-Temporal</span>
          </button>
          <button
            className={`preset-chip-btn ${activePreset === 'fusion' ? 'active' : ''}`}
            onClick={() => handleLoadPreset('fusion')}
            title="Two-Stream Optical + SAR Cross-Modal Fusion"
          >
            <Radio size={13} color="var(--amber-primary)" />
            <span>Optical + SAR</span>
          </button>
          <button
            className={`preset-chip-btn ${activePreset === 'multistep' ? 'active' : ''}`}
            onClick={() => handleLoadPreset('multistep')}
            title="Agentic Multi-Step Orchestration DAG"
          >
            <Sparkles size={13} color="var(--violet-primary)" />
            <span>Multi-Step</span>
          </button>
        </div>

        {/* Header Telemetry HUD */}
        <div className="header-telemetry">
          <div className="telemetry-badge">
            <span
              className="status-dot-pulse"
              style={{
                backgroundColor: backendStatus === 'connected' ? 'var(--emerald-primary)' : 'var(--rose-primary)',
                boxShadow: backendStatus === 'connected' ? '0 0 10px var(--emerald-primary)' : '0 0 10px var(--rose-primary)'
              }}
            />
            <span>{backendStatus === 'connected' ? 'SYS ONLINE' : 'OFFLINE'}</span>
          </div>
          <div className="telemetry-badge">
            <Cpu size={12} color="var(--cyan-primary)" />
            <span>LLM PLANNER</span>
          </div>
          <div className="telemetry-badge">
            <ShieldCheck size={12} color="var(--emerald-primary)" />
            <span>7/7 TOOLS</span>
          </div>
        </div>
      </header>

      {/* 3-Column Aerospace Workspace */}
      <main className="workspace-grid">
        <UploadPanel
          currentMode={currentMode}
          onModeChange={handleModeChange}
          uploadedImages={uploadedImages}
          onUploadComplete={handleUploadComplete}
          onClearImages={handleClearImages}
          presetLoading={presetLoading}
        />

        <MapViewer
          uploadedImages={uploadedImages}
          layers={layers}
          onLayerToggle={handleLayerToggle}
          loading={loading}
        />

        <ChatPanel
          currentMode={currentMode}
          uploadedImages={uploadedImages}
          onRunQuery={handleRunQuery}
          queryResponse={queryResponse}
          loading={loading}
          error={error}
          defaultPrompt={defaultPrompt}
        />
      </main>
    </div>
  );
}
