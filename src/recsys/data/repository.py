import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import os

from src.recsys.data.schemas import UserSchema, ProductSchema, InteractionSchema

class DataRepository(ABC):
    @abstractmethod
    def load_users(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_products(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_interactions(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_interactions(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def append_interaction(self, row: dict) -> None:
        pass

class CsvRepository(DataRepository):
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"

    def load_users(self) -> pd.DataFrame:
        path = self.raw_dir / "users.csv"
        df = pd.read_csv(path)
        return UserSchema.validate(df)

    def load_products(self) -> pd.DataFrame:
        path = self.raw_dir / "products.csv"
        df = pd.read_csv(path)
        return ProductSchema.validate(df)

    def load_interactions(self) -> pd.DataFrame:
        path = self.raw_dir / "interactions.csv"
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return InteractionSchema.validate(df)

    def save_interactions(self, df: pd.DataFrame) -> None:
        path = self.raw_dir / "interactions.csv"
        df = InteractionSchema.validate(df)
        df.to_csv(path, index=False)

    def append_interaction(self, row: dict) -> None:
        path = self.raw_dir / "interactions.csv"
        df_new = pd.DataFrame([row])
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"])
        InteractionSchema.validate(df_new)
        
        header = not path.exists()
        df_new.to_csv(path, mode='a', header=header, index=False)
