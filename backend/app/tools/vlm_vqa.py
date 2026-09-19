from typing import List, Type, Dict, Any
from pydantic import BaseModel
from .base import BaseTool
from ..schemas.tools import RSVQAInput, RSVQAOutput
from models.inference.vlm_wrapper import get_vlm_wrapper

class RSVQATool(BaseTool):
    """
    Real Remote Sensing VQA Tool powered by VLM inference wrapper.
    Answers natural language queries and extracts true token-probability confidence.
    """
    name: str = "rs_vqa"
    description: str = "Answers natural language questions regarding remote sensing imagery with calibrated confidence."
    input_schema: Type[BaseModel] = RSVQAInput
    output_schema: Type[BaseModel] = RSVQAOutput
    whitelisted_params: List[str] = ["query", "image_path", "confidence_threshold"]

    def __init__(self):
        self.vlm = get_vlm_wrapper()

    def run(self, input_data: RSVQAInput) -> RSVQAOutput:
        res = self.vlm.answer_vqa(
            query=input_data.query,
            image_path=input_data.image_path,
            confidence_threshold=input_data.confidence_threshold
        )
        return RSVQAOutput(
            answer=res["answer"],
            confidence=res["confidence"],
            evidence_summary=res["evidence"]
        )
