import os
import yaml
from pathlib import Path
from pydantic import BaseModel, field_validator
from typing import Dict, Any, List, Optional

class EventWeights(BaseModel):
    view: float
    click: float
    like: float
    purchase: float

class HybridWeights(BaseModel):
    cf: float
    content: float
    popularity: float
    
    @field_validator('cf', 'content', 'popularity')
    def check_non_negative(cls, v):
        if v < 0:
            raise ValueError('Weights must be non-negative')
        return v

class TfidfParams(BaseModel):
    stop_words: str
    ngram_range: List[int]
    min_df: int
    sublinear_tf: bool

class Paths(BaseModel):
    data_dir: str
    raw_dir: str
    processed_dir: str
    artifacts_dir: str

class LLMSettings(BaseModel):
    provider: str
    model: str
    timeout: float

class Settings(BaseModel):
    seed: int
    event_weights: EventWeights
    hybrid_weights: HybridWeights
    cold_start_threshold: int
    tfidf_params: TfidfParams
    paths: Paths
    llm_settings: LLMSettings
    top_k_default: int

def load_settings(config_path: str | Path = None) -> Settings:
    if config_path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "configs" / "default.yaml"
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        
    if "LLM_PROVIDER" in os.environ:
        config_dict.setdefault("llm_settings", {})["provider"] = os.environ["LLM_PROVIDER"]
    
    return Settings(**config_dict)

settings = load_settings()
