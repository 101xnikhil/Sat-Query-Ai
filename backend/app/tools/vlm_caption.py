from typing import List, Type
from pydantic import BaseModel
from .base import BaseTool
from ..schemas.tools import RSCaptionInput, RSCaptionOutput
from models.inference.vlm_wrapper import get_vlm_wrapper

class RSCaptionTool(BaseTool):
    """
    Remote Sensing Scene Captioning Tool powered by VLM inference wrapper.
    Generates structured descriptive captions and domain tags for remote sensing imagery.
    """
    name: str = "rs_caption"
    description: str = "Generates descriptive scene captions and domain tags from remote sensing imagery."
    input_schema: Type[BaseModel] = RSCaptionInput
    output_schema: Type[BaseModel] = RSCaptionOutput
    whitelisted_params: List[str] = ["image_path", "max_length", "style"]

    def __init__(self):
        self.vlm = get_vlm_wrapper()

    def run(self, input_data: RSCaptionInput) -> RSCaptionOutput:
        res = self.vlm.generate_caption(
            image_path=input_data.image_path,
            max_length=input_data.max_length,
            style=input_data.style
        )
        return RSCaptionOutput(
            caption=res["caption"],
            tags=res.get("tags", []),
            confidence=res["confidence"],
            rendering_applied=res.get("rendering_applied"),
            mode="model"
        )
