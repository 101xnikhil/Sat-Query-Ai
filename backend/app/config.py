import os
from pathlib import Path
from typing import Dict, Any, List
import yaml
from pydantic import BaseModel, Field

class ValidationSARConfig(BaseModel):
    db_conversion: bool = True
    epsilon: float = 1.0e-7
    speckle_filter: str = "lee"
    filter_window_size: int = 3
    damping_factor: float = 1.0

class ValidationConfig(BaseModel):
    min_overlap_iou: float = 0.10
    auto_reproject: bool = True
    default_target_crs: str = "EPSG:4326"
    sar: ValidationSARConfig = Field(default_factory=ValidationSARConfig)

class AppConfig(BaseModel):
    name: str = "SatQuery AI"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    storage_dir: str = "data/uploads"
    outputs_dir: str = "data/outputs"

class VLMConfig(BaseModel):
    name: str = "Qwen/Qwen2-VL-7B-Instruct"
    device: str = "cpu"
    lora_checkpoint_dir: str = "models/checkpoints/vqa_lora"
    max_tokens: int = 512

class ModelsConfig(BaseModel):
    base_vlm: VLMConfig = Field(default_factory=VLMConfig)
    datasets: Dict[str, Any] = Field(default_factory=dict)
    splits_dir: str = "data/datasets"

class ToolWhitelistItem(BaseModel):
    name: str
    allowed_params: List[str]

class ToolsConfig(BaseModel):
    whitelist: List[ToolWhitelistItem] = Field(default_factory=list)

class Settings(BaseModel):
    version: str = "1.0.0"
    app: AppConfig = Field(default_factory=AppConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

_settings: Settings = None

def get_settings(config_path: str = None) -> Settings:
    global _settings
    if _settings is not None and config_path is None:
        return _settings

    if config_path is None:
        # Check environment or default location
        env_path = os.getenv("SATQUERY_CONFIG_PATH")
        if env_path and os.path.exists(env_path):
            config_path = env_path
        else:
            # Look relative to project root
            base_dir = Path(__file__).resolve().parent.parent.parent
            for folder in ["configs", "config"]:
                default_path = base_dir / folder / "default_config.yaml"
                if default_path.exists():
                    config_path = str(default_path)
                    break

    data: Dict[str, Any] = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    _settings = Settings(**data)
    
    # Ensure storage and output directories exist
    base_dir = Path(__file__).resolve().parent.parent.parent
    storage_path = base_dir / _settings.app.storage_dir
    outputs_path = base_dir / _settings.app.outputs_dir
    storage_path.mkdir(parents=True, exist_ok=True)
    outputs_path.mkdir(parents=True, exist_ok=True)
    
    return _settings
