import React, { useEffect, useRef, useState } from 'react';
import * as maplibregl from 'maplibre-gl';
import { 
  Eye, 
  EyeOff, 
  Navigation, 
  Download, 
  Compass, 
  Maximize2, 
  Layers as LayersIcon,
  Crosshair,
  Globe
} from 'lucide-react';

export default function MapViewer({ uploadedImages, layers, onLayerToggle, loading }) {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [activeBasemap, setActiveBasemap] = useState('satellite');
  const [cursorCoords, setCursorCoords] = useState({ lat: 12.975, lon: 77.575, zoom: 12.0 });

  const basemaps = {
    satellite: {
      url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      attribution: 'Esri World Imagery'
    },
    dark: {
      url: 'https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
      attribution: 'CartoDB Dark Matter'
    },
    streets: {
      url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
      attribution: 'OpenStreetMap'
    }
  };

  // Initialize MapLibre GL map
  useEffect(() => {
    if (map.current) return;

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          'basemap-tiles': {
            type: 'raster',
            tiles: [basemaps[activeBasemap].url],
            tileSize: 256,
            attribution: basemaps[activeBasemap].attribution,
          },
        },
        layers: [
          {
            id: 'basemap-layer',
            type: 'raster',
            source: 'basemap-tiles',
            minzoom: 0,
            maxzoom: 19,
          },
        ],
      },
      center: [77.575, 12.975], // Bangalore default coordinates
      zoom: 11.5,
    });

    map.current.addControl(new maplibregl.NavigationControl({ showCompass: true }), 'bottom-right');

    map.current.on('load', () => {
      setMapLoaded(true);
    });

    map.current.on('mousemove', (e) => {
      setCursorCoords({
        lat: e.lngLat.lat.toFixed(4),
        lon: e.lngLat.lng.toFixed(4),
        zoom: map.current.getZoom().toFixed(1)
      });
    });

    return () => {
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, []);

  // Switch basemap source when activeBasemap changes
  useEffect(() => {
    if (!mapLoaded || !map.current) return;
    const source = map.current.getSource('basemap-tiles');
    if (source) {
      map.current.removeLayer('basemap-layer');
      map.current.removeSource('basemap-tiles');

      map.current.addSource('basemap-tiles', {
        type: 'raster',
        tiles: [basemaps[activeBasemap].url],
        tileSize: 256,
        attribution: basemaps[activeBasemap].attribution,
      });

      // Insert at bottom before overlay layers
      map.current.addLayer(
        {
          id: 'basemap-layer',
          type: 'raster',
          source: 'basemap-tiles',
          minzoom: 0,
          maxzoom: 19,
        },
        map.current.getLayer('img-footprint-fill') ? 'img-footprint-fill' : undefined
      );
    }
  }, [activeBasemap, mapLoaded]);

  // Fit bounds when new georeferenced images are uploaded
  useEffect(() => {
    if (!mapLoaded || !map.current || uploadedImages.length === 0) return;

    const firstGeo = uploadedImages.find((img) => img.is_georeferenced && img.bounds && img.bounds.crs === 'EPSG:4326');
    if (firstGeo && firstGeo.bounds) {
      const { minx, miny, maxx, maxy } = firstGeo.bounds;
      map.current.fitBounds(
        [[minx, miny], [maxx, maxy]],
        { padding: 70, duration: 1200 }
      );

      // Add image footprint rectangle
      const footprintId = 'img-footprint';
      if (map.current.getSource(footprintId)) {
        if (map.current.getLayer(`${footprintId}-fill`)) map.current.removeLayer(`${footprintId}-fill`);
        if (map.current.getLayer(`${footprintId}-line`)) map.current.removeLayer(`${footprintId}-line`);
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
          'fill-color': '#00f0ff',
          'fill-opacity': 0.08,
        },
      });

      map.current.addLayer({
        id: `${footprintId}-line`,
        type: 'line',
        source: footprintId,
        paint: {
          'line-color': '#00f0ff',
          'line-width': 1.5,
          'line-dasharray': [3, 2],
        },
      });
    }
  }, [uploadedImages, mapLoaded]);

  // Add / update GeoJSON layers on the map
  useEffect(() => {
    if (!mapLoaded || !map.current) return;

    layers.forEach((layer) => {
      const sourceId = `source-${layer.layer_id}`;
      const fillLayerId = `layer-fill-${layer.layer_id}`;
      const lineLayerId = `layer-line-${layer.layer_id}`;

      if (layer.geojson) {
        if (!map.current.getSource(sourceId)) {
          map.current.addSource(sourceId, {
            type: 'geojson',
            data: layer.geojson,
          });

          const color = layer.style?.color || '#00f0ff';
          const fillOpacity = layer.style?.fill_opacity !== undefined ? layer.style.fill_opacity : 0.45;

          map.current.addLayer({
            id: fillLayerId,
            type: 'fill',
            source: sourceId,
            paint: {
              'fill-color': color,
              'fill-opacity': fillOpacity,
            },
          });

          map.current.addLayer({
            id: lineLayerId,
            type: 'line',
            source: sourceId,
            paint: {
              'line-color': color,
              'line-width': 2,
            },
          });
        }

        // Toggle visibility
        const visibility = layer.visible !== false ? 'visible' : 'none';
        if (map.current.getLayer(fillLayerId)) {
          map.current.setLayoutProperty(fillLayerId, 'visibility', visibility);
        }
        if (map.current.getLayer(lineLayerId)) {
          map.current.setLayoutProperty(lineLayerId, 'visibility', visibility);
        }
      }
    });
  }, [layers, mapLoaded]);

  const handleFitExtent = () => {
    if (!map.current || uploadedImages.length === 0) return;
    const firstGeo = uploadedImages.find((img) => img.is_georeferenced && img.bounds);
    if (firstGeo && firstGeo.bounds) {
      const { minx, miny, maxx, maxy } = firstGeo.bounds;
      map.current.fitBounds([[minx, miny], [maxx, maxy]], { padding: 60, duration: 800 });
    }
  };

  return (
    <div className="map-canvas-container">
      {/* Sci-Fi Aerospace HUD Corner Brackets */}
      <div className="tactical-hud-bracket top-left" />
      <div className="tactical-hud-bracket top-right" />
      <div className="tactical-hud-bracket bottom-left" />
      <div className="tactical-hud-bracket bottom-right" />

      {/* Top-Left Telemetry Coordinates HUD */}
      <div className="map-hud-telemetry">
        <div className="hud-coord-item">
          <Crosshair size={12} color="var(--cyan-primary)" />
          <span>LAT:</span>
          <span className="hud-coord-value">{cursorCoords.lat}° N</span>
        </div>
        <div className="hud-coord-item">
          <span>LON:</span>
          <span className="hud-coord-value">{cursorCoords.lon}° E</span>
        </div>
        <div className="hud-coord-item">
          <span>ZOOM:</span>
          <span className="hud-coord-value">{cursorCoords.zoom}</span>
        </div>
        <div className="hud-coord-item">
          <span>CRS:</span>
          <span className="hud-coord-value">EPSG:4326</span>
        </div>
      </div>

      {/* Floating Tactical Layer & Basemap HUD (Top Right) */}
      <div className="map-floating-controls">
        <div className="floating-glass-card">
          <div className="card-title-header">
            <span>Tactical Basemap</span>
            <Globe size={13} color="var(--cyan-primary)" />
          </div>
          <div className="basemap-pill-group">
            <button
              className={`basemap-btn ${activeBasemap === 'satellite' ? 'active' : ''}`}
              onClick={() => setActiveBasemap('satellite')}
            >
              Satellite
            </button>
            <button
              className={`basemap-btn ${activeBasemap === 'dark' ? 'active' : ''}`}
              onClick={() => setActiveBasemap('dark')}
            >
              Tactical Dark
            </button>
            <button
              className={`basemap-btn ${activeBasemap === 'streets' ? 'active' : ''}`}
              onClick={() => setActiveBasemap('streets')}
            >
              Streets
            </button>
          </div>
        </div>

        {/* Tactical Overlay Layers HUD */}
        {layers.length > 0 && (
          <div className="floating-glass-card">
            <div className="card-title-header">
              <span>Active Overlays ({layers.length})</span>
              <button 
                onClick={handleFitExtent}
                title="Fit to bounds"
                style={{ background: 'transparent', border: 'none', color: 'var(--cyan-primary)', cursor: 'pointer' }}
              >
                <Maximize2 size={12} />
              </button>
            </div>
            <div>
              {layers.map((layer) => (
                <div key={layer.layer_id} className="layer-row-item">
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <span
                      className="layer-legend-dot"
                      style={{ backgroundColor: layer.style?.color || '#00f0ff', boxShadow: `0 0 6px ${layer.style?.color || '#00f0ff'}` }}
                    />
                    <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{layer.name}</span>
                  </div>
                  <button
                    onClick={() => onLayerToggle(layer.layer_id)}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: layer.visible !== false ? 'var(--cyan-primary)' : 'var(--text-muted)',
                      cursor: 'pointer'
                    }}
                  >
                    {layer.visible !== false ? <Eye size={13} /> : <EyeOff size={13} />}
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* High-Tech Animated Radar Scanline Overlay during Analysis */}
      {loading && <div className="radar-scan-overlay" />}

      {/* MapLibre DOM Node */}
      <div ref={mapContainer} className="maplibre-viewport" />
    </div>
  );
}
