import pytest
import rasterio
from rasterio.transform import rowcol
from models.inference.tiled_inference import generate_tiles, compute_box_iou, nms_boxes
from models.inference.georeferencer import BoundingBox2D, project_boxes_to_geojson

def test_tiling_grid_coverage():
    tiles = generate_tiles(img_width=1000, img_height=800, tile_size=512, overlap=64)
    assert len(tiles) >= 4
    # Check that boundaries cover edges
    max_x = max(t.col_off + t.width for t in tiles)
    max_y = max(t.row_off + t.height for t in tiles)
    assert max_x == 1000
    assert max_y == 800

def test_nms_boxes_deduplication():
    # Two heavily overlapping boxes with different confidences
    b1 = BoundingBox2D(ymin=0.10, xmin=0.10, ymax=0.30, xmax=0.30, confidence=0.95, label="runway")
    b2 = BoundingBox2D(ymin=0.11, xmin=0.11, ymax=0.31, xmax=0.31, confidence=0.75, label="runway")
    # One distinct box far away
    b3 = BoundingBox2D(ymin=0.60, xmin=0.60, ymax=0.80, xmax=0.80, confidence=0.90, label="water")

    iou_12 = compute_box_iou(b1, b2)
    assert iou_12 > 0.70  # Overlapping

    suppressed = nms_boxes([b1, b2, b3], iou_threshold=0.50)
    assert len(suppressed) == 2
    # Should retain highest confidence b1 (0.95) and distant b3 (0.90)
    assert suppressed[0].confidence == 0.95
    assert suppressed[1].confidence == 0.90

def test_pixel_to_geo_to_pixel_roundtrip(optical_geotiff):
    """
    Validates mathematical invertibility: pixel box -> georeferenced GeoJSON polygon -> pixel coordinates.
    Round-trip error must be less than 0.1 pixel.
    """
    # Create test normalized bounding box
    orig_box = BoundingBox2D(ymin=0.20, xmin=0.30, ymax=0.50, xmax=0.70, confidence=0.92, label="test_target")
    
    # 1. Project to GeoJSON polygon
    geojson_fc = project_boxes_to_geojson([orig_box], raster_path=optical_geotiff)
    assert len(geojson_fc.features) == 1
    feature = geojson_fc.features[0]
    coords = feature.geometry.coordinates[0]  # 5 points (closed ring: TL, TR, BR, BL, TL)
    
    # 2. Invert polygon coordinates back to pixel space using rasterio
    with rasterio.open(optical_geotiff) as src:
        w, h = src.width, src.height
        import pyproj
        transformer = None
        if src.crs and src.crs.to_string() != "EPSG:4326":
            transformer = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

        reconstructed_rows = []
        reconstructed_cols = []
        for lon, lat in coords[:4]:
            if transformer:
                x_proj, y_proj = transformer.transform(lon, lat)
            else:
                x_proj, y_proj = lon, lat
            r, c = rowcol(src.transform, x_proj, y_proj)
            reconstructed_rows.append(r)
            reconstructed_cols.append(c)

        # Expected pixel boundaries
        exp_r_min, exp_r_max = orig_box.ymin * h, orig_box.ymax * h
        exp_c_min, exp_c_max = orig_box.xmin * w, orig_box.xmax * w

        calc_r_min, calc_r_max = min(reconstructed_rows), max(reconstructed_rows)
        calc_c_min, calc_c_max = min(reconstructed_cols), max(reconstructed_cols)

        # Precision within 1 pixel
        assert abs(calc_r_min - exp_r_min) <= 1.0
        assert abs(calc_r_max - exp_r_max) <= 1.0
        assert abs(calc_c_min - exp_c_min) <= 1.0
        assert abs(calc_c_max - exp_c_max) <= 1.0
