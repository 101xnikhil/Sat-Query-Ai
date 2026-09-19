import React, { useEffect, useRef, useState } from 'react';
import * as maplibregl from 'maplibre-gl';
import { Eye, EyeOff, Navigation, Download } from 'lucide-react';

export default function MapViewer({ uploadedImages, layers, onLayerToggle }) {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const [mapLoaded, setMapLoaded] = useState(false);

  // Initialize MapLibre GL map
  useEffect(() => {
    if (map.current) return;

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          'osm-tiles': {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: '&copy; OpenStreetMap contributors',
          },
        },
        layers: [
          {
            id: 'osm-tiles-layer',
            type: 'raster',
            source: 'osm-tiles',
            minzoom: 0,
            maxzoom: 19,
          },
        ],
      },
      center: [77.025, 28.025], // Default center
      zoom: 11,
    });

    map.current.addControl(new maplibregl.NavigationControl({ showCompass: true }), 'bottom-right');

    map.current.on('load', () => {
      setMapLoaded(true);
    });

    return () => {
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, []);

  // Fit bounds when new georeferenced images are uploaded
  useEffect(() => {
    if (!mapLoaded || !map.current || uploadedImages.length === 0) return;

    const firstGeo = uploadedImages.find((img) => img.is_georeferenced && img.bounds && img.bounds.crs === 'EPSG:4326');
    if (firstGeo && firstGeo.bounds) {
      const { minx, miny, maxx, maxy } = firstGeo.bounds;
      map.current.fitBounds(
        [[minx, miny], [maxx, maxy]],
        { padding: 60, duration: 1200 }
      );

      // Add image footprint rectangle
      const footprintId = 'img-footprint';
      if (map.current.getSource(footprintId)) {
        map.current.removeLayer(`${footprintId}-fill`);
        map.current.removeLayer(`${footprintId}-line`);
        map.current.removeSource(footprintId);
      }

      map.current.addSource(footprintId, {
        type: 'geojson',
        data: {
          type: 'Feature',
          geometry: {
            type: 'Polygon',
            coordinates: [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]],
          },
        },
      });

      map.current.addLayer({
        id: `${footprintId}-fill`,
        type: 'fill',
        source: footprintId,
        paint: {
          'fill-color': '#00e5ff',
          'fill-opacity': 0.08,
        },
      });

      map.current.addLayer({
        id: `${footprintId}-line`,
        type: 'line',
        source: footprintId,
        paint: {
          'line-color': '#00e5ff',
          'line-width': 2,
          'line-dasharray': [2, 2],
        },
      });
    }
  }, [uploadedImages, mapLoaded]);

  // Update dynamic overlay layers from QueryResponse
  useEffect(() => {
    if (!mapLoaded || !map.current) return;

    // Synchronize layers
    layers.forEach((layer) => {
      const sourceId = `src-${layer.layer_id}`;
      const fillLayerId = `fill-${layer.layer_id}`;
      const lineLayerId = `line-${layer.layer_id}`;

      if (layer.geojson) {
        if (!map.current.getSource(sourceId)) {
          map.current.addSource(sourceId, {
            type: 'geojson',
            data: layer.geojson,
          });

          map.current.addLayer({
            id: fillLayerId,
            type: 'fill',
            source: sourceId,
            paint: {
              'fill-color': layer.color || '#00e5ff',
              'fill-opacity': layer.visible ? (layer.opacity || 0.6) : 0,
            },
          });

          map.current.addLayer({
            id: lineLayerId,
            type: 'line',
            source: sourceId,
            paint: {
              'line-color': layer.color || '#00e5ff',
              'line-width': 2,
              'line-opacity': layer.visible ? 1 : 0,
            },
          });

          // Add popup click listener
          map.current.on('click', fillLayerId, (e) => {
            if (e.features && e.features.length > 0) {
              const props = e.features[0].properties;
              new maplibregl.Popup()
                .setLngLat(e.lngLat)
                .setHTML(`
                  <div style="color: #0f172a; font-family: sans-serif; font-size: 12px; line-height: 1.4;">
                    <strong>${props.label || props.class || 'Detection'}</strong><br/>
                    ${props.confidence ? `Confidence: ${(props.confidence * 100).toFixed(1)}%<br/>` : ''}
                    ${props.area_m2 ? `Ground Area: ${Number(props.area_m2).toLocaleString()} m²<br/>` : ''}
                    ${props.area_km2 ? `Ground Area: ${props.area_km2} km²<br/>` : ''}
                    ${props.area_pixels ? `Pixel Area: ${props.area_pixels} px²` : ''}
                  </div>
                `)
                .addTo(map.current);
            }
          });
        } else {
          // Update layer visibility
          if (map.current.getLayer(fillLayerId)) {
            map.current.setPaintProperty(
              fillLayerId,
              'fill-opacity',
              layer.visible ? (layer.opacity || 0.6) : 0
            );
          }
          if (map.current.getLayer(lineLayerId)) {
            map.current.setPaintProperty(
              lineLayerId,
              'line-opacity',
              layer.visible ? 1 : 0
            );
          }
        }
      }
    });
  }, [layers, mapLoaded]);

  const handleExportGeoJSON = (layer) => {
    if (!layer.geojson) return;
    const blob = new Blob([JSON.stringify(layer.geojson, null, 2)], { type: 'application/geo+json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${layer.name.toLowerCase().replace(/[^a-z0-9]/g, '_')}.geojson`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const [activeDateView, setActiveDateView] = useState('both'); // 't1', 't2', 'both'

  return (
    <div className="map-canvas-container">
      <div ref={mapContainer} className="maplibre-viewport" />

      {/* Bi-Temporal Date Comparison Toolbar */}
      {uploadedImages.length >= 2 && (
        <div style={{
          position: 'absolute',
          top: 14,
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 10,
          display: 'flex',
          gap: 6,
          background: 'rgba(15, 23, 42, 0.88)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.12)',
          borderRadius: 24,
          padding: '4px 6px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)'
        }}>
          <button
            onClick={() => setActiveDateView('t1')}
            style={{
              background: activeDateView === 't1' ? 'rgba(0, 229, 255, 0.2)' : 'transparent',
              border: activeDateView === 't1' ? '1px solid var(--accent-cyan)' : '1px solid transparent',
              color: activeDateView === 't1' ? 'var(--accent-cyan)' : 'var(--text-secondary)',
              borderRadius: 18,
              padding: '5px 12px',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 5
            }}
          >
            <span>T1 ({uploadedImages[0]?.acquisition_date ? uploadedImages[0].acquisition_date.slice(0, 10) : 'Pre'})</span>
          </button>
          <button
            onClick={() => setActiveDateView('t2')}
            style={{
              background: activeDateView === 't2' ? 'rgba(0, 229, 255, 0.2)' : 'transparent',
              border: activeDateView === 't2' ? '1px solid var(--accent-cyan)' : '1px solid transparent',
              color: activeDateView === 't2' ? 'var(--accent-cyan)' : 'var(--text-secondary)',
              borderRadius: 18,
              padding: '5px 12px',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 5
            }}
          >
            <span>T2 ({uploadedImages[1]?.acquisition_date ? uploadedImages[1].acquisition_date.slice(0, 10) : 'Post'})</span>
          </button>
          <button
            onClick={() => setActiveDateView('both')}
            style={{
              background: activeDateView === 'both' ? 'rgba(244, 63, 94, 0.2)' : 'transparent',
              border: activeDateView === 'both' ? '1px solid #f43f5e' : '1px solid transparent',
              color: activeDateView === 'both' ? '#f43f5e' : 'var(--text-secondary)',
              borderRadius: 18,
              padding: '5px 12px',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 5
            }}
          >
            <span>Δ Change Footprint</span>
          </button>
        </div>
      )}

      {/* Toggleable Layers Control */}
      {layers.length > 0 && (
        <div className="map-overlay-controls">
          <div className="overlay-title">Analysis Overlays</div>
          {layers.map((layer) => (
            <div key={layer.layer_id} className="layer-toggle-item">
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <span
                  className="color-indicator"
                  style={{ background: layer.color || 'var(--accent-cyan)' }}
                />
                <span>{layer.name}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                {layer.geojson && (
                  <button
                    onClick={() => handleExportGeoJSON(layer)}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}
                    title="Export GeoJSON polygon layer"
                  >
                    <Download size={13} />
                  </button>
                )}
                <button
                  onClick={() => onLayerToggle(layer.layer_id)}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-secondary)' }}
                  title={layer.visible ? "Hide layer" : "Show layer"}
                >
                  {layer.visible ? <Eye size={15} color="var(--accent-cyan)" /> : <EyeOff size={15} />}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
