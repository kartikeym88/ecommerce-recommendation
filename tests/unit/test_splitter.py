import pandas as pd
from src.recsys.evaluation.splitter import TemporalSplitter

def test_splitter():
    events = []
    for i in range(10):
        events.append({
            "user_id": 1,
            "product_id": i,
            "event_type": "purchase" if i == 9 else "view",
            "rating": None,
            "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)
        })
    df = pd.DataFrame(events)
    
    splitter = TemporalSplitter()
    train_df, test_df = splitter.split(df, test_fraction=0.2, min_train=5)
    
    assert len(test_df) == 2
    assert len(train_df) == 8
    
    assert train_df["timestamp"].max() < test_df["timestamp"].min()
    assert "purchase" in test_df["event_type"].values
