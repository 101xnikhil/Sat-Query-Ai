import React, { useState, useRef } from 'react';
import { Upload, Layers, Image as ImageIcon, Calendar, Radio, CheckCircle, AlertCircle } from 'lucide-react';
import { uploadImagery } from '../services/api';

export default function UploadPanel({ onUploadComplete, currentMode, onModeChange, uploadedImages }) {
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
        // If 2 files dropped, upload both
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

  return (
    <div className="panel">
      <div className="panel-header">
        <span>IMAGERY INGESTION</span>
        <Layers size={16} color="var(--accent-cyan)" />
      </div>

      <div className="panel-body">
        {/* Mode Selector */}
        <div>
          <label className="overlay-title" style={{ display: 'block', marginBottom: 6 }}>
            Input Configuration
          </label>
          <div className="mode-tabs">
            <button
              className={`mode-tab-btn ${currentMode === 'single' ? 'active' : ''}`}
              onClick={() => onModeChange('single')}
            >
              Single Scene
            </button>
            <button
              className={`mode-tab-btn ${currentMode === 'optical_sar' ? 'active' : ''}`}
              onClick={() => onModeChange('optical_sar')}
            >
              Optical + SAR
            </button>
            <button
              className={`mode-tab-btn ${currentMode === 'bitemporal' ? 'active' : ''}`}
              onClick={() => onModeChange('bitemporal')}
            >
              Bi-Temporal
            </button>
          </div>
        </div>

        {/* Bi-temporal Acquisition Dates */}
        {currentMode === 'bitemporal' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div>
              <label style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: 2 }}>
                T1 Date (Epoch 1)
              </label>
              <input
                type="date"
                value={t1Date}
                onChange={(e) => setT1Date(e.target.value)}
                style={{
                  width: '100%',
                  background: 'rgba(0,0,0,0.3)',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-primary)',
                  padding: '6px',
                  borderRadius: '6px',
                  fontSize: '0.75rem',
                }}
              />
            </div>
            <div>
              <label style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: 2 }}>
                T2 Date (Epoch 2)
              </label>
              <input
                type="date"
                value={t2Date}
                onChange={(e) => setT2Date(e.target.value)}
                style={{
                  width: '100%',
                  background: 'rgba(0,0,0,0.3)',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-primary)',
                  padding: '6px',
                  borderRadius: '6px',
                  fontSize: '0.75rem',
                }}
              />
            </div>
          </div>
        )}

        {/* Dropzone */}
        <div
          className="dropzone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            style={{ display: 'none' }}
            multiple={currentMode !== 'single'}
            accept=".tif,.tiff,.png,.jpg,.jpeg"
            onChange={(e) => handleFiles(e.target.files)}
          />
          <Upload size={28} className="dropzone-icon" />
          <div className="dropzone-text">
            {uploading ? 'Processing & Validating...' : currentMode === 'single' ? 'Upload Single Raster' : 'Upload Pair of Images'}
          </div>
          <div className="dropzone-sub">
            Supports GeoTIFF / TIFF & PNG/JPEG benchmarks
          </div>
        </div>

        {error && (
          <div style={{ color: 'var(--accent-rose)', fontSize: '0.78rem', display: 'flex', gap: 6, alignItems: 'center' }}>
            <AlertCircle size={14} />
            <span>{error}</span>
          </div>
        )}

        {/* Ingested Images Metadata List */}
        <div>
          <div className="overlay-title">Ingested Rasters ({uploadedImages.length})</div>
          {uploadedImages.length === 0 ? (
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textAlign: 'center', padding: '16px 0' }}>
              No images uploaded yet. Upload sample or benchmark data to begin.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {uploadedImages.map((img, idx) => (
                <div key={img.image_id || idx} className="image-meta-card">
                  <div className="meta-title">
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '200px' }}>
                      {img.filename}
                    </span>
                    <span className="meta-tag">{img.modality}</span>
                  </div>
                  <div className="meta-grid">
                    <div className="meta-item">Dims: <span>{img.width}x{img.height}</span></div>
                    <div className="meta-item">Bands: <span>{img.band_count} ({img.dtype})</span></div>
                    <div className="meta-item">CRS: <span>{img.crs || 'Pixel Grid'}</span></div>
                    <div className="meta-item">GeoRef: <span>{img.is_georeferenced ? 'Yes (WGS84)' : 'No'}</span></div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
