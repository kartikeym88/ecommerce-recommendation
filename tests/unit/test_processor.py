import pandas as pd
from src.recsys.preprocessing.processor import DataProcessor

def test_processor():
    users = pd.DataFrame([{"user_id": 1}])
    products = pd.DataFrame([{"product_id": 1}])
    interactions = pd.DataFrame([
        {"interaction_id": 1, "user_id": 1, "product_id": 1, "event_type": "view", "rating": None, "timestamp": pd.Timestamp.now()},
        {"interaction_id": 2, "user_id": 1, "product_id": 1, "event_type": "like", "rating": None, "timestamp": pd.Timestamp.now()}
    ])
    
    proc = DataProcessor(event_weights={"view": 1.0, "like": 3.0})
    processed = proc.run(users, products, interactions)
    
    strength_df = processed.strength_df
    assert len(strength_df) == 1
    assert abs(strength_df.iloc[0]["strength"] - 1.60943) < 0.01
