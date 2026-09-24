import pandas as pd
from abc import ABC, abstractmethod
from src.recsys.domain import Evidence
from src.recsys.preprocessing.matrix import InteractionMatrix

class BaseRecommender(ABC):
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, matrix: InteractionMatrix, products: pd.DataFrame):
        pass
        
    @abstractmethod
    def score_user(self, user_id: int) -> pd.Series:
        pass
        
    @abstractmethod
    def similar_items(self, product_id: int, n: int) -> list[int]:
        pass
        
    @abstractmethod
    def explain(self, user_id: int, product_id: int) -> Evidence:
        pass
