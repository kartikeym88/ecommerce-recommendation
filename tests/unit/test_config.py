import os
import pytest
from src.recsys.config import load_settings, Settings
import yaml
from pydantic import ValidationError

def test_load_default_settings():
    settings = load_settings()
    assert isinstance(settings, Settings)
    assert settings.seed == 42
    assert settings.hybrid_weights.cf == 0.6

def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "test_provider")
    settings = load_settings()
    assert settings.llm_settings.provider == "test_provider"

def test_invalid_weights(tmp_path):
    config_dict = {
        "seed": 42,
        "event_weights": {"view": 1.0, "click": 2.0, "like": 3.0, "purchase": 5.0},
        "hybrid_weights": {"cf": -0.5, "content": 0.3, "popularity": 0.1},
        "cold_start_threshold": 5,
        "tfidf_params": {"stop_words": "english", "ngram_range": [1, 2], "min_df": 1, "sublinear_tf": True},
        "paths": {"data_dir": "data", "raw_dir": "data/raw", "processed_dir": "data/processed", "artifacts_dir": "artifacts"},
        "llm_settings": {"provider": "anthropic", "model": "test", "timeout": 1.0},
        "top_k_default": 10
    }
    
    config_file = tmp_path / "test.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)
        
    with pytest.raises(ValidationError):
        load_settings(config_file)
