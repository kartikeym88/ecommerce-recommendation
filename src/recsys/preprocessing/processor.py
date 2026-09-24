import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.recsys.config import settings

@dataclass
class ProcessedData:
    strength_df: pd.DataFrame
    
class DataProcessor:
    def __init__(self, event_weights=None):
        self.event_weights = event_weights or settings.event_weights.model_dump()
        
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(subset=["user_id", "product_id"])
        
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["interaction_id"])
        now = pd.Timestamp.now()
        df = df[df["timestamp"] <= now].copy()
        
        if "rating" in df.columns:
            df["rating"] = df["rating"].clip(1.0, 5.0)
            
        return df
        
    def to_implicit_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["weight"] = df["event_type"].map(self.event_weights).fillna(0.0)
        
        if "rating" in df.columns:
            rating_adj = np.where(df["rating"].isna(), 0.0,
                         np.where(df["rating"] >= 4.0, 2.0,
                         np.where(df["rating"] <= 2.0, -2.0, 0.0)))
            df["weight"] += rating_adj
            
        grouped = df.groupby(["user_id", "product_id"])["weight"].sum().reset_index()
        grouped["strength"] = np.log1p(grouped["weight"].clip(lower=0.0))
        return grouped
        
    def run(self, users: pd.DataFrame, products: pd.DataFrame, interactions: pd.DataFrame) -> ProcessedData:
        valid = self.validate(interactions)
        cleaned = self.clean(valid)
        
        cleaned = cleaned[cleaned["user_id"].isin(users["user_id"]) & cleaned["product_id"].isin(products["product_id"])]
        
        strength_df = self.to_implicit_strength(cleaned)
        return ProcessedData(strength_df=strength_df)
