from backend.app.controller.router import RuleBasedRouter
from backend.app.schemas.common import Modality, TaskType, ImageMetadata, RasterFormat, GeoBBox

def make_dummy_meta(image_id: str, modality: Modality, path: str = "test.tif") -> ImageMetadata:
    return ImageMetadata(
        image_id=image_id,
        filename="test.tif",
        file_path=path,
        format=RasterFormat.GEOTIFF,
        modality=modality,
        width=100,
        height=100,
        bounds=GeoBBox(minx=77.0, miny=28.0, maxx=77.05, maxy=28.05),
        crs="EPSG:4326"
    )

def test_router_single_vqa():
    router = RuleBasedRouter()
    images = [make_dummy_meta("img1", Modality.OPTICAL)]
    task = router.classify_task("What is the primary land use in this scene?", images)
    assert task == TaskType.SINGLE_VQA
    plan = router.plan_tools(task, "What is the primary land use in this scene?", images)
    assert plan[0][0] == "rs_vqa"

def test_router_grounding():
    router = RuleBasedRouter()
    images = [make_dummy_meta("img1", Modality.OPTICAL)]
    task = router.classify_task("Where are the airport runways located?", images)
    assert task == TaskType.GROUNDING
    plan = router.plan_tools(task, "Where are the airport runways located?", images)
    assert plan[0][0] == "rs_grounding"

def test_router_caption():
    router = RuleBasedRouter()
    images = [make_dummy_meta("img1", Modality.OPTICAL)]
    task = router.classify_task("Describe the landscape and features of this satellite image", images)
    assert task == TaskType.CAPTION
    plan = router.plan_tools(task, "Describe the landscape and features of this satellite image", images)
    assert plan[0][0] == "rs_caption"

def test_router_optical_sar_analysis():
    router = RuleBasedRouter()
    images = [make_dummy_meta("opt", Modality.OPTICAL), make_dummy_meta("sar", Modality.SAR)]
    task = router.classify_task("Segment water and built-up areas using both optical and radar", images)
    assert task == TaskType.OPTICAL_SAR_ANALYSIS
    plan = router.plan_tools(task, "Segment water and built-up areas", images)
    assert plan[0][0] == "optical_sar_fusion"
    assert plan[1][0] == "spectral_index"

def test_router_change_vqa():
    router = RuleBasedRouter()
    images = [make_dummy_meta("t1", Modality.OPTICAL), make_dummy_meta("t2", Modality.OPTICAL)]
    task = router.classify_task("What changed in the forest canopy between 2021 and 2023?", images)
    assert task == TaskType.CHANGE_VQA
    plan = router.plan_tools(task, "What changed in the forest canopy?", images)
    assert plan[0][0] == "change_map"
    assert plan[1][0] == "change_vqa"
