import pytest
from models.inference.georeferencer import project_boxes_to_geojson, BoundingBox2D
from backend.app.schemas.common import GeoJSONFeatureCollection

def test_project_boxes_geotiff(optical_geotiff):
    boxes = [
        BoundingBox2D(ymin=0.1, xmin=0.1, ymax=0.4, xmax=0.4, label="water_test", confidence=0.92, normalized=True),
        BoundingBox2D(ymin=0.6, xmin=0.6, ymax=0.8, xmax=0.8, label="urban_test", confidence=0.88, normalized=True)
    ]
    geojson = project_boxes_to_geojson(boxes, optical_geotiff, query="Locate water and urban")
    assert isinstance(geojson, GeoJSONFeatureCollection)
    assert len(geojson.features) == 2

    feat1 = geojson.features[0]
    assert feat1.properties["label"] == "water_test"
    assert feat1.properties["confidence"] == 0.92
    assert feat1.properties["georeferenced"] is True

    # Check geographic coordinates fall inside Delhi area (77.0 to 77.05, 28.0 to 28.05)
    coords = feat1.geometry.coordinates[0]
    for pt in coords:
        lon, lat = pt[0], pt[1]
        assert 77.0 <= lon <= 77.05
        assert 28.0 <= lat <= 28.05

def test_project_boxes_utm_reproject(utm_reproject_geotiff):
    boxes = [
        BoundingBox2D(ymin=0.2, xmin=0.2, ymax=0.5, xmax=0.5, label="runway", confidence=0.95, normalized=True)
    ]
    geojson = project_boxes_to_geojson(boxes, utm_reproject_geotiff)
    assert len(geojson.features) == 1
    # Check transformed coordinates are in lat/long range (~28 N, 77 E)
    coords = geojson.features[0].geometry.coordinates[0]
    for pt in coords:
        lon, lat = pt[0], pt[1]
        assert 76.0 <= lon <= 78.0
        assert 27.0 <= lat <= 29.0

def test_project_boxes_non_georeferenced_png(benchmark_png):
    boxes = [
        BoundingBox2D(ymin=0.25, xmin=0.25, ymax=0.75, xmax=0.75, label="box_pixel", confidence=0.85, normalized=True)
    ]
    geojson = project_boxes_to_geojson(boxes, benchmark_png)
    assert len(geojson.features) == 1
    feat = geojson.features[0]
    assert feat.properties["georeferenced"] is False
    # Pixel coords on 128x128 image
    coords = feat.geometry.coordinates[0]
    assert coords[0] == [32.0, 32.0]
    assert coords[2] == [96.0, 96.0]
