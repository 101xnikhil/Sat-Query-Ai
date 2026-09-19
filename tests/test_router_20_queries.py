import pytest
from backend.app.schemas.common import Modality, TaskType
from backend.ingest.reader import read_geotiff_metadata
from backend.controller.router import RuleBasedRouter, IncompatibleInputError

# 24 diverse natural language remote sensing queries across all 6 tasks
QUERIES_AND_EXPECTED_TASKS = [
    # Task 1: Single VQA (Optical / Multispectral or SAR single image)
    ("What is the dominant land cover or terrain in this scene?", TaskType.SINGLE_VQA, 1, False),
    ("Is there an active river or body of water visible?", TaskType.SINGLE_VQA, 1, False),
    ("How many runways are present at the airfield?", TaskType.SINGLE_VQA, 1, False),
    ("What type of agricultural crops are growing here?", TaskType.SINGLE_VQA, 1, False),
    ("What is the vegetation index NDVI for this area?", TaskType.SINGLE_VQA, 1, False),
    
    # Task 2: Caption (Single image scene description and tagging)
    ("Describe the satellite imagery scene in detail.", TaskType.CAPTION, 1, False),
    ("Provide an overview summary of this urban area.", TaskType.CAPTION, 1, False),
    ("Give a concise scene summary and key land use tags.", TaskType.CAPTION, 1, False),
    ("What is this scene showing in terms of terrain and buildings?", TaskType.CAPTION, 1, False),

    # Task 3: Grounding (Single image object detection / bounding coordinates)
    ("Where are the aircraft parked on the tarmac?", TaskType.GROUNDING, 1, False),
    ("Locate the perimeter of the central water reservoir.", TaskType.GROUNDING, 1, False),
    ("Find bounding boxes for all commercial buildings.", TaskType.GROUNDING, 1, False),
    ("Show me the coordinates of the primary bridges.", TaskType.GROUNDING, 1, False),
    ("Detect and highlight all storage tanks in the port area.", TaskType.GROUNDING, 1, False),

    # Task 4: Change VQA (Bi-temporal pair question answering)
    ("What changed between the two acquisition dates?", TaskType.CHANGE_VQA, 2, False),
    ("How did the water reservoir surface area change between dates?", TaskType.CHANGE_VQA, 2, False),
    ("Did urban construction expand between image 1 and image 2?", TaskType.CHANGE_VQA, 2, False),
    ("Is there any noticeable deforestation or vegetation loss?", TaskType.CHANGE_VQA, 2, False),

    # Task 5: Change Map (Bi-temporal change mask and area quantification)
    ("Generate a change map showing spatial extent of change.", TaskType.CHANGE_MAP, 2, False),
    ("How much area changed in square kilometers between T1 and T2?", TaskType.CHANGE_MAP, 2, False),
    ("Create a binary change mask for the flooded zones.", TaskType.CHANGE_MAP, 2, False),
    ("Calculate the percentage changed between the two dates.", TaskType.CHANGE_MAP, 2, False),

    # Task 6: Optical-SAR Fusion (Cross-modal complementary segmentation)
    ("Fuse optical and SAR imagery to segment water bodies.", TaskType.OPTICAL_SAR_ANALYSIS, 2, True),
    ("Perform cross-modal optical and SAR built-up area extraction.", TaskType.OPTICAL_SAR_ANALYSIS, 2, True),
    ("Map flooded regions by fusing radar backscatter and optical reflectance.", TaskType.OPTICAL_SAR_ANALYSIS, 2, True),
]

@pytest.fixture
def router():
    return RuleBasedRouter()

@pytest.mark.parametrize("query,expected_task,img_count,is_optical_sar", QUERIES_AND_EXPECTED_TASKS)
def test_router_20_queries_classification(
    router,
    optical_geotiff,
    sar_geotiff,
    query,
    expected_task,
    img_count,
    is_optical_sar
):
    opt_meta = read_geotiff_metadata(optical_geotiff)
    sar_meta = read_geotiff_metadata(sar_geotiff)

    if img_count == 1:
        images = [opt_meta]
    elif is_optical_sar:
        images = [opt_meta, sar_meta]
    else:
        # Bi-temporal optical pair
        opt2 = opt_meta.model_copy()
        opt2.image_id = "opt2"
        images = [opt_meta, opt2]

    task = router.classify_task(query, images)
    assert task == expected_task, f"Query '{query}' classified as {task}, expected {expected_task}"

    # Verify tool plan generation
    plan = router.plan_tools(task, query, images)
    assert len(plan) >= 1
    tool_names = [p[0] for p in plan]
    assert all(isinstance(t, str) for t in tool_names)

def test_router_incompatible_change_single_image(router, optical_geotiff):
    """Change query provided with only a single image must be rejected."""
    opt_meta = read_geotiff_metadata(optical_geotiff)
    with pytest.raises(IncompatibleInputError, match="requires exactly 2 images"):
        router.classify_task("What changed between the two dates?", [opt_meta])

def test_router_incompatible_change_map_single_image(router, optical_geotiff):
    """Change map query provided with only a single image must be rejected."""
    opt_meta = read_geotiff_metadata(optical_geotiff)
    with pytest.raises(IncompatibleInputError, match="requires exactly 2 images"):
        router.classify_task("Generate a change map and calculate area changed", [opt_meta])

def test_router_incompatible_fusion_single_image(router, optical_geotiff):
    """Optical-SAR fusion query provided with only a single image must be rejected."""
    opt_meta = read_geotiff_metadata(optical_geotiff)
    with pytest.raises(IncompatibleInputError, match="requires exactly 2 images"):
        router.classify_task("Fuse optical and SAR imagery to delineate flood boundaries", [opt_meta])

def test_router_incompatible_dual_image_for_caption(router, optical_geotiff):
    """Two optical images provided for a single-image caption request without change context must be rejected."""
    opt1 = read_geotiff_metadata(optical_geotiff)
    opt2 = opt1.model_copy()
    with pytest.raises(IncompatibleInputError, match="requires exactly 1 image"):
        router.classify_task("Describe this satellite scene in detail", [opt1, opt2])
