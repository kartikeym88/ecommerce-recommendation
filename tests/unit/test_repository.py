import pandas as pd
import pytest
from pathlib import Path
from src.recsys.data.repository import CsvRepository
import pandera.errors

@pytest.fixture
def repo(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    return CsvRepository(tmp_path)

def test_repository_validates_data(repo):
    df_bad = pd.DataFrame([{"user_id": 1}])
    df_bad.to_csv(repo.raw_dir / "users.csv", index=False)
    
    with pytest.raises(pandera.errors.SchemaError):
        repo.load_users()

def test_append_interaction(repo):
    row = {
        "interaction_id": 1,
        "user_id": 1,
        "product_id": 1,
        "event_type": "view",
        "rating": None,
        "timestamp": "2026-09-24T10:14:00"
    }
    repo.append_interaction(row)
    
    df = repo.load_interactions()
    assert len(df) == 1
    assert df.iloc[0]["event_type"] == "view"
