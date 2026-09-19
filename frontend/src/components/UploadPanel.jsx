import React, { useState, useRef } from 'react';
import { 
  Upload, 
  Layers, 
  Image as ImageIcon, 
  Calendar, 
  Radio, 
  CheckCircle, 
  AlertCircle,
  Trash2,
  Maximize2,
  FileCode,
  Sliders
} from 'lucide-react';
import { uploadImagery } from '../services/api';

export default function UploadPanel({ 
  onUploadComplete, 
  currentMode, 
  onModeChange, 
  uploadedImages,
  onClearImages,
  presetLoading
}) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [t1Date, setT1Date] = useState('2023-01-15');
  const [t2Date, setT2Date] = useState('2024-01-15');
  const fileInputRef = useRef(null);

  const handleFiles = async (files) => {
    if (!files || files.length === 0) return;
    setUploading(true);
    setError(null);

    try {
      let res;
      if (currentMode === 'single') {
        res = await uploadImagery([files[0]]);
      } else if (currentMode === 'optical_sar') {
        res = await uploadImagery(Array.from(files));
      } else if (currentMode === 'bitemporal') {
        res = await uploadImagery(Array.from(files), null, `${t1Date}T00:00:00Z`);
      }
      onUploadComplete(res.images);
    } catch (err) {
      setError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
  };

  const onDragOver = (e) => {
    e.preventDefault();
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <span>IMAGERY INGESTION DECK</span>
        <span className="panel-header-badge">GDAL / RASTERIO</span>
      </div>

      <div className="panel-body">
        {/* Input Configuration Selector */}
        <div>
          <div className="card-title-header">
            <span>Mission Configuration</span>
            <span style={{ fontSize: '0.68rem', color: 'var(--cyan-primary)' }}>AUTO-VALIDATED</span>
          </div>
          <div className="mode-tabs">
            <button
              className={`mode-tab-btn ${currentMode === 'single' ? 'active' : ''}`}
              onClick={() => onModeChange('single')}
            >
              <span>Single Optical</span>
              <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)' }}>VQA / Grounding</span>
            </button>
            <button
              className={`mode-tab-btn ${currentMode === 'optical_sar' ? 'active' : ''}`}
              onClick={() => onModeChange('optical_sar')}
            >
              <span>Optical + SAR</span>
              <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)' }}>Cross-Modal Fusion</span>
            </button>
            <button
              className={`mode-tab-btn ${currentMode === 'bitemporal' ? 'active' : ''}`}
              onClick={() => onModeChange('bitemporal')}
            >
              <span>Bi-Temporal</span>
              <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)' }}>Change Vector RCVA</span>
            </button>
          </div>
        </div>

        {/* Bi-Temporal Observation Epoch Dates */}
        {currentMode === 'bitemporal' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, background: 'rgba(0,0,0,0.3)', padding: 10, borderRadius: 10, border: '1px solid var(--border-subtle)' }}>
            <div>
              <label style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4 }}>
                <Calendar size={11} color="var(--cyan-primary)" />
                <span>T1 Date (Epoch 1)</span>
              </label>
              <input
                type="date"
                value={t1Date}
                onChange={(e) => setT1Date(e.target.value)}
                style={{
                  width: '100%',
                  background: 'rgba(15, 23, 42, 0.8)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 6,
                  color: 'var(--text-primary)',
                  fontSize: '0.78rem',
                  padding: '5px 8px',
                  fontFamily: 'var(--font-mono)'
                }}
              />
            </div>
            <div>
              <label style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4 }}>
                <Calendar size={11} color="var(--rose-primary)" />
                <span>T2 Date (Epoch 2)</span>
              </label>
              <input
                type="date"
                value={t2Date}
                onChange={(e) => setT2Date(e.target.value)}
                style={{
                  width: '100%',
                  background: 'rgba(15, 23, 42, 0.8)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 6,
                  color: 'var(--text-primary)',
                  fontSize: '0.78rem',
                  padding: '5px 8px',
                  fontFamily: 'var(--font-mono)'
                }}
              />
            </div>
          </div>
        )}

        {/* Drag & Drop Ingestion Zone */}
        <div
          className="dropzone"
          onDrop={onDrop}
          onDragOver={onDragOver}
          onClick={() => fileInputRef.current && fileInputRef.current.click()}
        >
          <input
            type="file"
            multiple={currentMode !== 'single'}
            accept=".tif,.tiff,.png,.jpg,.jpeg"
            ref={fileInputRef}
            style={{ display: 'none' }}
            onChange={(e) => handleFiles(e.target.files)}
          />
          <Upload size={28} className="dropzone-icon" />
          <div className="dropzone-text">
            {uploading ? 'Ingesting & Validating Raster...' : 'Drop Satellite Imagery Here'}
          </div>
          <div className="dropzone-sub">
            Click to browse or drop {currentMode === 'single' ? '1 scene' : '2+ co-registered scenes'}
          </div>
          <div className="dropzone-formats">
            <span className="format-chip">GeoTIFF</span>
            <span className="format-chip">TIFF</span>
            <span className="format-chip">EPSG:4326</span>
            <span className="format-chip">UTM</span>
            <span className="format-chip">MAX 200MB</span>
          </div>
        </div>

        {error && (
          <div style={{ padding: '8px 12px', borderRadius: 8, background: 'rgba(244, 63, 94, 0.15)', border: '1px solid rgba(244, 63, 94, 0.3)', color: 'var(--rose-primary)', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: 6 }}>
            <AlertCircle size={14} />
            <span>{error}</span>
          </div>
        )}

        {/* Ingested Raster Telemetry Cards */}
        <div>
          <div className="card-title-header">
            <span>Ingested Rasters ({uploadedImages.length})</span>
            {uploadedImages.length > 0 && (
              <button 
                onClick={onClearImages}
                style={{ background: 'transparent', border: 'none', color: 'var(--rose-primary)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}
              >
                <Trash2 size={11} />
                <span>CLEAR</span>
              </button>
            )}
          </div>

          {uploadedImages.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '24px 12px', background: 'rgba(0,0,0,0.2)', borderRadius: 10, border: '1px dashed var(--border-subtle)', color: 'var(--text-muted)', fontSize: '0.76rem' }}>
              No imagery loaded yet. Drop GeoTIFF files or select a Capability above to auto-load presets.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {uploadedImages.map((img, idx) => {
                const isSAR = img.modality === 'sar';
                return (
                  <div key={img.image_id || idx} className="image-meta-card">
                    <div className="meta-header-row">
                      <span className="meta-filename" title={img.filename}>
                        {img.filename}
                      </span>
                      <span className={`modality-pill ${isSAR ? 'modality-sar' : 'modality-optical'}`}>
                        {isSAR ? 'RADAR / SAR' : 'OPTICAL / MS'}
                      </span>
                    </div>

                    <div className="meta-details-grid">
                      <div className="meta-stat-item">
                        <span>CRS:</span>
                        <span className="meta-stat-value">{img.crs || 'Local Grid'}</span>
                      </div>
                      <div className="meta-stat-item">
                        <span>Dimensions:</span>
                        <span className="meta-stat-value">{img.width} × {img.height}</span>
                      </div>
                      <div className="meta-stat-item">
                        <span>Bands:</span>
                        <span className="meta-stat-value">{img.band_count} ({img.dtype})</span>
                      </div>
                      <div className="meta-stat-item">
                        <span>Resolution:</span>
                        <span className="meta-stat-value">
                          {img.resolution && img.resolution.length > 0 ? `${(img.resolution[0] * 111320).toFixed(1)}m` : '10m'}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
